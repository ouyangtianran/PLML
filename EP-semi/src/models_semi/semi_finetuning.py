"""
Few-Shot Parallel: trains a model as a series of tasks computed in parallel on multiple GPUs

"""
import numpy as np
import os
import torch
import torch.nn.functional as F

from src.tools.meters import BasicMeter
from src.modules.distances import prototype_distance
from embedding_propagation import EmbeddingPropagation, LabelPropagation
from .base_wrapper import BaseWrapper
from haven import haven_utils as haven
from scipy.stats import sem, t
import shutil as sh
import sys

from src.utils import pairwise_distances
from .utils import get_proto, get_pseudo_label, top_accuracy, remove_prefix

from utils.semco.label_embedding_guessor import LabelEmbeddingGuessor
from utils.semco.labels2wv import get_grouping, get_labels2wv_dict
from utils.semco.parser import parse_args
import torch.nn as nn


class FinetuneWrapper(BaseWrapper):
    """Finetunes a model using an episodic scheme on multiple GPUs"""

    def __init__(self, model, nclasses, exp_dict, pretrained_embs= None, n_word_emb=300):
        """ Constructor
        Args:
            model: architecture to train
            nclasses: number of output classes
            exp_dict: reference to dictionary with the hyperparameters
        """
        super().__init__()
        self.model = model
        self.exp_dict = exp_dict
        self.ngpu = self.exp_dict["ngpu"]

        self.embedding_propagation = EmbeddingPropagation()
        self.label_propagation = LabelPropagation()
        self.model.add_classifier(nclasses, modalities=0)
        self.nclasses = nclasses

        # self.pretrained_embs = torch.tensor(pretrained_embs, dtype=torch.float32).cuda(non_blocking=True)

        # self.model.add_adjust(self.model.out_size, n_word_emb)
        dropout = self.exp_dict["adjust_dropout"]
        type = self.exp_dict["adjust_type"]

        self.model.add_adjust(self.model.output_size, 1, name='self_attention', type=type, dropout=dropout)
        self.model.add_adjust(n_word_emb, self.model.output_size, name='adjust_word', type=type, dropout=dropout)
        self.model.add_sigmoid()

        if self.exp_dict["rotation_weight"] > 0:
            self.model.add_classifier(4, "classifier_rot")

        # semco
        # define criterion for upper(semantic embedding) and lower(discrete label) paths
        self.crit_lower = lambda inp, targ: F.cross_entropy(inp, targ, reduction='none')
        self.crit_upper = lambda inp, targ: 1 - F.cosine_similarity(inp, targ)
        self.semco_config = parse_args()
        self.semco_config.lambda_emb = exp_dict["lambda_emb"]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.add_adjust(self.model.output_size, n_word_emb, name='adjust_emb', type=type, dropout=dropout)
        # add transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model.output_size,
            nhead=8,
            dim_feedforward=self.model.output_size,
            dropout=0.1,
            activation="relu",
        )
        trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        setattr(self.model, 'trans_encoder', trans_encoder)

        best_accuracy = -1
        if self.exp_dict["pretrained_weights_root"] is not None:
            pretrain_models_flag = 'saved_models'
            if pretrain_models_flag in self.exp_dict["pretrained_weights_root"]:
                pm = self.exp_dict["pretrain_method"]
                model_name = self.exp_dict['model']["backbone"]
                dataset_name = self.exp_dict['dataset_train_root']
                npc = self.exp_dict['num_per_class']
                if pm == 'semco':
                    pth_name = f'{pm}_{model_name}_{dataset_name}_npc{npc}'.lower()
                else:
                    pth_name = '{}_{}_{}_npc{:03d}'.format(pm,dataset_name,model_name,npc).lower()
                for model_pth in os.listdir(self.exp_dict['pretrained_weights_root']):
                    if pth_name in model_pth:
                        if pm == 'semco':
                            base_path = os.path.join(self.exp_dict['pretrained_weights_root'], model_pth)
                            # model_static_dict = torch.load(base_path)['model_state_dict']
                            model_static_dict = torch.load(base_path)['ema_shadow']
                            model_static_dict = remove_prefix(model_static_dict)
                        elif pm == 'flexmatch':
                            base_path = os.path.join(self.exp_dict['pretrained_weights_root'], model_pth, 'model_best.pth')
                            # model_static_dict = torch.load(base_path)['model']
                            model_static_dict = torch.load(base_path)['ema_model']
                            model_static_dict = remove_prefix(model_static_dict)
                        self.model.load_state_dict(model_static_dict,
                                                   strict=False)
                        print('load model: ' + model_pth)
                        best_accuracy = 0.2
                        break
                assert (best_accuracy > 0.1)
            else:
                for exp_hash in os.listdir(self.exp_dict['pretrained_weights_root']):
                    base_path = os.path.join(self.exp_dict['pretrained_weights_root'], exp_hash)
                    exp_dict_path = os.path.join(base_path, 'exp_dict.json')
                    if not os.path.exists(exp_dict_path):
                        continue
                    loaded_exp_dict = haven.load_json(exp_dict_path)
                    if "unlabeled_multiple" not in loaded_exp_dict.keys():
                        continue
                    pkl_path = os.path.join(base_path, 'score_list_best.pkl')
                    if (loaded_exp_dict["model"]["name"] == 'semi_pretraining' and
                            loaded_exp_dict["dataset_train"].split('_')[-1] == exp_dict["dataset_train"].split('_')[-1] and
                            loaded_exp_dict["model"]["backbone"] == exp_dict['model']["backbone"] and
                            loaded_exp_dict["num_per_class"] == exp_dict["num_per_class"] and
                            loaded_exp_dict["word_emb_weight"] == exp_dict["word_emb_weight"] and
                            # loaded_exp_dict["unlabeled_size_train"] == exp_dict["unlabeled_size_train"] and
                            # loaded_exp_dict["unlabeled_multiple"] == exp_dict["unlabeled_multiple0"] and
                            loaded_exp_dict["transform_val"] == exp_dict["transform_val"] and
                            # loaded_exp_dict["labelprop_alpha"] == exp_dict["labelprop_alpha"] and
                            # loaded_exp_dict["labelprop_scale"] == exp_dict["labelprop_scale"] and
                            os.path.exists(pkl_path)):
                        accuracy = haven.load_pkl(pkl_path)[-1]["val_accuracy"]
                        print(exp_hash)
                        try:
                            self.model.load_state_dict(torch.load(os.path.join(base_path, 'checkpoint_best.pth'))['model'],
                                                       strict=False)
                            if accuracy > best_accuracy:
                                best_path = os.path.join(base_path, 'checkpoint_best.pth')
                                best_accuracy = accuracy
                        except:
                            continue
                assert (best_accuracy > 0.1)
                print("Finetuning %s with original accuracy : %f" % (best_path, best_accuracy))
                self.model.load_state_dict(torch.load(best_path)['model'], strict=False)

        # Add optimizers here
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.exp_dict["lr"],
                                         momentum=0.9,
                                         weight_decay=self.exp_dict["weight_decay"],
                                         nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode="min" if "loss" in self.exp_dict[
                                                                        "target_loss"] else "max",
                                                                    patience=self.exp_dict["patience"])
        # optimizers for model with frozen parameters
        if self.exp_dict["freeze"]:
            self.freeze_model()
            self.print_freeze_layers()
        self.optimizer_freeze = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                        lr=self.exp_dict["lr"],
                        momentum=0.9,
                        weight_decay=self.exp_dict["weight_decay"],
                        nesterov=True)
        self.scheduler_freeze = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_freeze,
                                                                    mode="min" if "loss" in self.exp_dict[
                                                                        "target_loss"] else "max",
                                                                    patience=self.exp_dict["patience"])
        if self.exp_dict["freeze"]:
            self.unfreeze_model()

        self.model.cuda()
        if self.ngpu > 1:
            self.parallel_model = torch.nn.DataParallel(self.model, device_ids=list(range(self.ngpu)))

    def _get_label_guessor(self):
        classes = self.class_names
        class_2_embeddings_dict = self.embs_dict
        # get_labels2wv_dict(classes, self.semco_config.word_vec_path)
        emb_dim = len(list(class_2_embeddings_dict.values())[0])
        if self.semco_config.eps is None:
            eps = 0.15 if emb_dim < 100 else 0.2 if emb_dim < 256 else 0.28  # for label grouping clustering
        else:
            eps = self.semco_config.eps
        label_group_idx, gr_mapping = get_grouping(class_2_embeddings_dict, eps=eps, return_mapping=True)
        label_guessor = LabelEmbeddingGuessor(classes, label_group_idx, class_2_embeddings_dict, self.semco_config.thr_emb,
                                              self.device)
        return label_guessor, emb_dim

    def get_logits(self, embeddings, support_size, query_size, nclasses):
        """Computes the logits from the queries of an episode
        
        Args:
            embeddings (torch.Tensor): episode embeddings
            support_size (int): size of the support set
            query_size (int): size of the query set
            nclasses (int): number of classes
        
        Returns:
            torch.Tensor: logits
        """
        b, c = embeddings.size()

        propagator = None
        # if self.exp_dict["embedding_prop"] == True:
        #     embeddings = self.embedding_propagation(embeddings)

        if self.exp_dict["distance_type"] == "labelprop":
            support_labels = torch.arange(nclasses, device=embeddings.device).view(1, nclasses).repeat(support_size,
                                                                                                       1).view(
                support_size, nclasses)
            unlabeled_labels = nclasses * torch.ones(query_size * nclasses, dtype=support_labels.dtype,
                                                     device=support_labels.device).view(query_size, nclasses)
            labels = torch.cat([support_labels, unlabeled_labels], 0).view(-1)
            n_labels = len(labels)
            if b > n_labels:
                unlabeled_labels2 = nclasses * torch.ones(b-n_labels, dtype=support_labels.dtype,
                                                     device=support_labels.device).view(-1)
                labels = torch.cat([labels, unlabeled_labels2], 0).view(-1)
            logits = self.label_propagation(embeddings, labels, nclasses)
            logits1 = logits[:n_labels].view(-1, nclasses, nclasses)[support_size:(support_size + query_size), ...].view(-1,
                                                                                                             nclasses)
            if b > n_labels:
                logits2 = logits[n_labels:]
                logits = torch.cat([logits1, logits2],0)
            else:
                logits = logits1

        elif self.exp_dict["distance_type"] == "prototypical":
            embeddings = embeddings.view(-1, nclasses, c)
            support_embeddings = embeddings[:support_size].view(-1, c)
            query_embeddings = embeddings[support_size:].view(-1, c)
            support_labels = torch.arange(nclasses, device=embeddings.device).view(1, nclasses).repeat(support_size,
                                                                                                       1).view(-1)
            logits = prototype_distance(support_embeddings, query_embeddings,
                                        support_labels).view(query_size * nclasses, nclasses)
        return logits

    def get_word_logits(self, embeddings, support_size, query_size, nclasses, word_embs):
        """Computes the logits from the queries of an episode

        Args:
            embeddings (torch.Tensor): episode embeddings
            support_size (int): size of the support set
            query_size (int): size of the query set
            nclasses (int): number of classes

        Returns:
            torch.Tensor: logits
        """
        b, c = embeddings.size()

        propagator = None
        # if self.exp_dict["embedding_prop"] == True:
        #     embeddings = self.embedding_propagation(embeddings)
        # embeddings = embeddings.view(-1, nclasses, c)
        #
        # support_embeddings = word_embs
        # query_embeddings = embeddings
        # query_size = support_size + query_size
        # support_size  = 1
        #
        # logits = prototype_distance((support_embeddings.view(1, support_size, nclasses, c), False),
        #                             (query_embeddings.view(1, query_size, nclasses, c), False)).view(
        #     query_size * nclasses, nclasses)
        logits = - pairwise_distances(embeddings.view(-1, c), word_embs.view(-1, c), 'l2')

        # if self.exp_dict["distance_type"] == "labelprop":
        #     support_labels = torch.arange(nclasses, device=embeddings.device).view(1, nclasses).repeat(support_size,
        #                                                                                                1).view(
        #         support_size, nclasses)
        #     unlabeled_labels = nclasses * torch.ones(query_size * nclasses, dtype=support_labels.dtype,
        #                                              device=support_labels.device).view(query_size, nclasses)
        #     labels = torch.cat([support_labels, unlabeled_labels], 0).view(-1)
        #     logits = self.label_propagation(embeddings, labels, nclasses)
        #     logits = logits.view(-1, nclasses, nclasses)[support_size:(support_size + query_size), ...].view(-1,
        #                                                                                                      nclasses)
        #
        # elif self.exp_dict["distance_type"] == "prototypical":
        #     embeddings = embeddings.view(-1, nclasses, c)
        #     support_embeddings = embeddings[:support_size]
        #     query_embeddings = embeddings[support_size:]
        #     logits = prototype_distance((support_embeddings.view(1, support_size, nclasses, c), False),
        #                                 (query_embeddings.view(1, query_size, nclasses, c), False)).view(
        #         query_size * nclasses, nclasses)
        return logits

    def get_semco_loss(self, embeddings, y, embeddings_un, k,n,k_u,n_u):
        b,c = embeddings.size()
        if self.exp_dict["embedding_prop"] == True:
            embeddings_all = torch.cat([embeddings, embeddings_un])
            embeddings_all = self.embedding_propagation(embeddings_all)
            embeddings = embeddings_all[:b]
            embeddings_un = embeddings_all[b:]
        embeddings = embeddings.view(k,n,c)
        embeddings_un = embeddings_un.view(k_u,n_u,c)
        emb__x, emb_x_strong = embeddings[:,0,...].view(-1,c), embeddings[:,n-1,...].view(-1,c)
        emb_u_w, emb_u_s = embeddings_un[:, 0, ...].view(-1,c), embeddings_un[:, n - 1, ...].view(-1,c)
        lbs_x = y.view(k,n)[:,0].view(-1)
        logits_x = self.model.classifier(emb__x)
        logits_u_w, logits_u_s = self.model.classifier(emb_u_w), self.model.classifier(emb_u_s)
        # get embedding
        logits_emb__x = self.model.adjust_emb(emb__x)
        logits_emb_u_w, logits_emb_u_s = self.model.adjust_emb(emb_u_w), self.model.adjust_emb(emb_u_s)

        criteria_x = self.crit_lower
        criteria_u = self.crit_lower
        criteria_x_emb = self.crit_upper
        criteria_u_emb = self.crit_upper
        # supervised loss for upper and lower paths
        loss_x = criteria_x(logits_x, lbs_x).mean()
        loss_x_emb = criteria_x_emb(logits_emb__x, self.label_emb_guessor.embedding_matrix[lbs_x]).mean()

        # guessing the labels for upper and lower paths
        with torch.no_grad():
            probs = torch.softmax(logits_u_w, dim=1)
            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(self.semco_config.thr).float()
            # get label guesses and mask based on embedding predictions (upper path)
            lbs_emb_u_guess, mask_emb, scores_emb, lbs_guess_help = self.label_emb_guessor(logits_emb_u_w)

        # combining the losses via co-training (blind version)
        mask_combined = mask.bool() | mask_emb.bool()
        # each loss path will have two components (co-training implementation)
        loss_u = (criteria_u(logits_u_s, lbs_u_guess) * mask).mean() + \
                 (criteria_u(logits_u_s, lbs_guess_help) * mask_emb).mean() * (self.semco_config.lambda_emb) / 3

        loss_u_emb = (criteria_u_emb(logits_emb_u_s, lbs_emb_u_guess) * mask_emb).mean() + \
                     (criteria_u_emb(logits_emb_u_s,
                                     self.label_emb_guessor.embedding_matrix[lbs_u_guess]) * mask).mean()

        loss_lower = loss_x + self.semco_config.lam_u * loss_u
        loss_upper = loss_x_emb + self.semco_config.lam_u * loss_u_emb
        loss = loss_lower + self.semco_config.lambda_emb * loss_upper

        # loss = 0
        return loss

    def freeze_model(self):
        for k, v in self.model.named_parameters():
            if 'conv' in k or 'bn' in k:
                ss = k.split('.')
                nn = int(ss[0][-1])
                if nn < self.exp_dict["freeze"]:
                    v.requires_grad = False
                    # print('freeze {}'.format(k))

    def print_freeze_layers(self):
        ss = ""
        for k, v in self.model.named_parameters():
            if v.requires_grad == False:
                ss += k
                ss += " "
        if ss=="":
            print('no layer is frozen')
        else:
            print('freeze layers: {}'.format(ss))

    def unfreeze_model(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def train_on_batch(self, batch):
        """Computes the loss on an episode
        
        Args:
            batch (dict): Episode dict
        
        Returns:
            tuple: loss and accuracy of the episode 
        """
        episode = batch[0]
        nclasses = episode["nclasses"]
        support_size = episode["support_size"]
        query_size = episode["query_size"]
        labels = episode["targets"].view(support_size + query_size, nclasses, -1).cuda(non_blocking=True).long()
        labels_weight = episode["targets_weight"].view(support_size + query_size, nclasses, -1).cuda(non_blocking=True).long()
        query_labels_weight = labels_weight[support_size:, ...].contiguous().view(-1)
        k = (support_size + query_size)
        c = episode["channels"]
        h = episode["height"]
        w = episode["width"]

        tx = episode["support_set"].view(support_size, nclasses, c, h, w).cuda(non_blocking=True)
        vx = episode["query_set"].view(query_size, nclasses, c, h, w).cuda(non_blocking=True)
        x = torch.cat([tx, vx], 0)
        x = x.view(-1, c, h, w).cuda(non_blocking=True)
        if self.ngpu > 1:
            embeddings = self.parallel_model(x, is_support=True)
        else:
            embeddings = self.model(x, is_support=True)
        # adjust visual embeddings to word embedding space
        # embeddings = self.model.adjust(embeddings)
        u_labels = labels[0]
        n_labels = torch.numel(labels)
        # unlabeled data
        # if episode["unlabeled_set"] is not None:
        #     ux = episode["unlabeled_set"].view(-1, c, h, w).cuda(non_blocking=True)
        #     if self.ngpu > 1:
        #         embeddings_un = self.parallel_model(ux, is_support=True)
        #     else:
        #         embeddings_un = self.model(ux, is_support=True)
        #     with torch.no_grad():
        #         y = labels.view(-1)
        #         proto = get_proto(self.pretrained_embs, embeddings, y)
        #         l_un, y_u, embeddings_un = get_pseudo_label(embeddings_un, proto,
        #                                                 y=y, remove_noise=self.exp_dict['remove_noise'], y_u_gt=episode["un_labels"])
        #     if len(y_u) > 0:
        #         labels = torch.cat([y, y_u],0)
        #         embeddings = torch.cat([embeddings, embeddings_un],0)
        emb0 = embeddings

        if self.exp_dict["trans_encoder"]:
            # adjust embedding with transformer encoder
            embeddings = self.model.trans_encoder(torch.unsqueeze(embeddings, dim=1)).squeeze()
            if "trans_encoder_cls" in self.exp_dict.keys():
                if self.exp_dict["trans_encoder_cls"]:
                    emb0 = embeddings

        b, c = embeddings.size()

        loss = 0
        if "embedding_prop_clc" in self.exp_dict.keys():
            if self.exp_dict["embedding_prop_clc"] == True:
                embeddings = self.embedding_propagation(embeddings)

        if self.exp_dict["classification_weight"] > 0:
            if "weight_loss" in self.exp_dict.keys():
                if not self.exp_dict["weight_loss"]:
                    loss += F.cross_entropy(self.model.classifier(emb0.view(b, c)), labels.view(-1)) * self.exp_dict[
                    "classification_weight"]
            else:
                loss_weight = F.cross_entropy(self.model.classifier(emb0.view(b, c)), labels.view(-1), reduction='none') * labels_weight
                loss += loss_weight.mean() * self.exp_dict["classification_weight"]

        # if self.exp_dict["word_emb_weight"] > 0:
        #     # with torch.autograd.set_detect_anomaly(True):
        #     emb = episode["embs"]
        #     k, n, emb_c = emb.size()
        #     emb = emb.view(-1, emb_c)
        #     emb = emb[:support_size * nclasses].cuda(non_blocking=True)
        #     # emb = emb.cuda(non_blocking=True)
        #     # emb = self.model.adjust_word(emb)
        #     # lambda_c = self.model.self_attention(emb)
        #     # lambda_c = self.model.sigmoid(lambda_c)
        #     lambda_c = 0.5
        #     # expand
        #     # emb = emb.view(1, nclasses, -1).repeat(support_size, 1, 1).view(support_size * nclasses, -1)
        #     # lambda_c = lambda_c.view(1, nclasses).repeat(support_size, 1).view(support_size * nclasses, 1)
        #     # with torch.no_grad():
        #     sup_embs = embeddings[:support_size * nclasses]
        #     qur_embs = embeddings[support_size * nclasses:]
        #     sup_embs = emb * lambda_c + sup_embs * (1 - lambda_c)
        #     embeddings = torch.cat([sup_embs, qur_embs], 0)

        if self.exp_dict["embedding_prop"] == True and self.exp_dict["embedding_prop_clc"] == False:
            embeddings = self.embedding_propagation(embeddings)
        logits = self.get_logits(embeddings, support_size, query_size, nclasses)

        query_labels = torch.arange(nclasses, device=logits.device).view(1, nclasses).repeat(query_size, 1).view(-1)
        # if b > n_labels:
        #     y_u1 = y_u
        #     for i, l in enumerate(u_labels):
        #         y_u1[y_u==l] = i
        #     query_labels = torch.cat([query_labels, y_u1],0)
        if "weight_loss" in self.exp_dict.keys():
            if not self.exp_dict["weight_loss"]:
                loss += F.cross_entropy(logits, query_labels) * self.exp_dict["few_shot_weight"]
        else:
            loss_weight = F.cross_entropy(logits, query_labels, reduction='none') * query_labels_weight
            loss += loss_weight.mean() * self.exp_dict["few_shot_weight"]

        # if self.exp_dict["word_emb_weight"] > 0:
        #     embs = episode["embs"].cuda(non_blocking=True)
        #     logits = self.get_word_logits(embeddings, support_size, query_size, nclasses, embs)
        #
        #     query_labels = torch.arange(nclasses, device=logits.device).view(1, nclasses).repeat(query_size + support_size, 1).view(-1)
        #     loss += F.cross_entropy(logits, query_labels) * self.exp_dict["word_emb_weight"]
        return loss

    def train_no_episodic_on_batch(self, batch, batch_un):
        """Computes the loss of a batch

        Args:
            batch (tuple): Inputs and labels

        Returns:
            loss: Loss on the batch
        """
        x, y, r, ims = batch
        x_u, y_u, r_u, ims = batch_un
        y = y.cuda(non_blocking=True).view(-1)
        r = r.cuda(non_blocking=True).view(-1)
        k, n, c, h, w = x.size()
        x = x.view(n * k, c, h, w).cuda(non_blocking=True)
        if self.ngpu > 1:
            embeddings = self.parallel_model(x, is_support=True)
        else:
            embeddings = self.model(x, is_support=True)

        b, c = embeddings.size()

        loss = 0
        # if self.exp_dict["unsupervised_loss"] == 'rotation':
        if self.exp_dict["rotation_weight"] > 0:
            rot = self.model.classifier_rot(embeddings)
            loss += F.cross_entropy(rot, r) * self.exp_dict["rotation_weight"]

        # if self.exp_dict["word_emb_weight"] > 0:
        #     emb = emb.view(n * k, -1).cuda(non_blocking=True)
        #     distance = embeddings - emb
        #     distance = (distance ** 2).mean(dim=1)
        #     loss += distance.mean() * self.exp_dict["word_emb_weight"]

        # unlabeled data loss
        if self.exp_dict["unlabeled_multiple"] > 0:
            r_u = r_u.cuda(non_blocking=True).view(-1)
            k_u, n_u, c, h, w = x_u.size()

            x_u = x_u.view(n_u * k_u, c, h, w).cuda(non_blocking=True)
            if self.ngpu > 1:
                embeddings_un = self.parallel_model(x_u, is_support=True)
            else:
                embeddings_un = self.model(x_u, is_support=True)
            if self.exp_dict["unsupervised_loss"] == 'simsiam':
                b, c = embeddings_un.size()
                embeddings_un = embeddings_un.view(k_u, n_u, c)
                loss += self.get_simsiam_loss(embeddings_un, c)
                # loss for labeled data
                if self.exp_dict["embedding_prop"] == True:
                    embeddings = self.embedding_propagation(embeddings)
                logits = self.model.classifier(embeddings)
                loss += F.cross_entropy(logits, y) * self.exp_dict["cross_entropy_weight"]
            elif self.exp_dict["unsupervised_loss"] == 'semco':
                if self.exp_dict["rotation_weight"] > 0:
                    rot = self.model.classifier_rot(embeddings_un)
                    loss += F.cross_entropy(rot, r_u) * self.exp_dict["rotation_weight"]
                loss += self.get_semco_loss(embeddings, y, embeddings_un,
                                            k, n, k_u, n_u)

            else:
                if self.exp_dict["rotation_weight"] > 0:
                    rot = self.model.classifier_rot(embeddings_un)
                    loss += F.cross_entropy(rot, r_u) * self.exp_dict["rotation_weight"]
                if self.exp_dict["unsupervised_loss"] == 'word_proto':
                    # get pseudo label
                    proto = get_proto(self.pretrained_embs, embeddings, y)
                    l_un, y_u, embeddings_un = get_pseudo_label(embeddings_un, proto,
                                                                remove_noise=self.exp_dict['remove_noise'],
                                                                y=y)
                    with torch.no_grad():
                        w_u = torch.exp(l_un / self.T)
                        w_u_m, _ = torch.max(w_u, dim=1)
                        w_u = w_u_m / torch.sum(w_u, dim=1)
                        w_y = torch.ones_like(y)
                        # w_u = torch.ones_like(y_u)
                        w_u = w_u.detach()
                        y_u = y_u.detach()
                    # unsupervised loss
                    # concat embeddings and embedding_un
                    embeddings = torch.cat([embeddings, embeddings_un])
                    y = torch.cat([y, y_u])
                    w_y = torch.cat([w_y, w_u])
                    if self.exp_dict["embedding_prop"] == True:
                        embeddings = self.embedding_propagation(embeddings)
                    logits = self.model.classifier(embeddings)
                    loss0 = F.cross_entropy(logits, y, reduction='none') * self.exp_dict["cross_entropy_weight"]
                    loss += (loss0 * w_y).mean()
                    # loss += F.cross_entropy(logits, y) * self.exp_dict["cross_entropy_weight"]
                elif self.exp_dict["unsupervised_loss"] == 'classifier_loss':
                    loss_clc = self.get_clc_loss(embeddings_un, embeddings, y)
                    loss += loss_clc

                # if self.exp_dict["embedding_prop"] == True:
                #     embeddings = self.embedding_propagation(embeddings)
                # logits = self.model.classifier(embeddings)
                # loss += F.cross_entropy(logits, y) * self.exp_dict["cross_entropy_weight"]
        else:  # only use label dataset
            if "embedding_prop_clc" in self.exp_dict.keys():
                if self.exp_dict["embedding_prop_clc"] == True:
                    embeddings = self.embedding_propagation(embeddings)

            logits = self.model.classifier(embeddings)
            # logits = torch.einsum('nc,ck->nk', [embeddings, proto.T])
            # # apply temperature
            # logits /= self.T
            loss += F.cross_entropy(logits, y) * self.exp_dict["cross_entropy_weight"]

        return loss

    def predict_on_batch(self, batch):
        """Computes the logits of an episode
        
        Args:
            batch (dict): episode dict
        
        Returns:
            tensor: logits for the queries of the current episode
        """
        nclasses = batch["nclasses"]
        support_size = batch["support_size"]
        query_size = batch["query_size"]
        k = (support_size + query_size)
        c = batch["channels"]
        h = batch["height"]
        w = batch["width"]

        tx = batch["support_set"].view(support_size, nclasses, c, h, w).cuda(non_blocking=True)
        vx = batch["query_set"].view(query_size, nclasses, c, h, w).cuda(non_blocking=True)
        x = torch.cat([tx, vx], 0)
        x = x.view(-1, c, h, w).cuda(non_blocking=True)

        if self.ngpu > 1:
            embeddings = self.parallel_model(x, is_support=True)
        else:
            embeddings = self.model(x, is_support=True)
        # adjust visual embeddings to word embedding space
        # embeddings = self.model.adjust(embeddings)

        # if self.exp_dict["word_emb_weight"] > 0:
        #     # with torch.autograd.set_detect_anomaly(True):
        #     emb = batch["embs"]
        #     k, n, emb_c = emb.size()
        #     emb = emb.view(-1, emb_c)
        #     emb = emb[:support_size * nclasses].cuda(non_blocking=True)
        #     # emb = emb.cuda(non_blocking=True)
        #     # emb = self.model.adjust_word(emb)
        #     # lambda_c = self.model.self_attention(emb)
        #     # lambda_c = self.model.sigmoid(lambda_c)
        #     lambda_c = 0.5
        #     # expand
        #     # emb = emb.view(1, nclasses, -1).repeat(support_size, 1, 1).view(support_size * nclasses, -1)
        #     # lambda_c = lambda_c.view(1, nclasses).repeat(support_size, 1).view(support_size * nclasses, 1)
        #     embeddings[:support_size * nclasses] = emb * lambda_c + embeddings[:support_size * nclasses] * (
        #             1 - lambda_c)
        if self.exp_dict["trans_encoder"]:
            embeddings = self.model.trans_encoder(torch.unsqueeze(embeddings, dim=1)).squeeze()

        if self.exp_dict["embedding_prop"] == True:
            embeddings = self.embedding_propagation(embeddings)

        return self.get_logits(embeddings, support_size, query_size, nclasses)

    def train_test_on_batch(self, batch):
        """Computes the loss and accuracy on a validation batch

        Args:
            batch (dict): Episode dict

        Returns:
            tuple: loss and accuracy of the episode
        """
        x, y, r, ims = batch
        y = y.cuda(non_blocking=True).view(-1)
        r = r.cuda(non_blocking=True).view(-1)
        k, n, c, h, w = x.size()
        x = x.view(n * k, c, h, w).cuda(non_blocking=True)
        if self.ngpu > 1:
            embeddings = self.parallel_model(x, is_support=True)
        else:
            embeddings = self.model(x, is_support=True)
        if self.exp_dict["trans_encoder"] and "trans_encoder_cls" in self.exp_dict.keys():
            if self.exp_dict["trans_encoder_cls"]:
                embeddings = self.model.trans_encoder(torch.unsqueeze(embeddings, dim=1)).squeeze()

        if "embedding_prop_clc" in self.exp_dict.keys():
            if self.exp_dict["embedding_prop_clc"] == True:
                embeddings = self.embedding_propagation(embeddings)

        logits = self.model.classifier(embeddings)
        loss = F.cross_entropy(logits, y)
        acc = float(logits.max(-1)[1].eq(y).float().mean())
        # scores = torch.softmax(logits, dim=1)
        # top1, top5 = top_accuracy(scores, y, (1, 5))

        return loss, acc

    def val_on_batch(self, batch):
        """Computes the loss and accuracy on a validation batch
        
        Args:
            batch (dict): Episode dict
        
        Returns:
            tuple: loss and accuracy of the episode 
        """
        nclasses = batch["nclasses"]
        query_size = batch["query_size"]

        logits = self.predict_on_batch(batch)

        query_labels = torch.arange(nclasses, device=logits.device).view(1, nclasses).repeat(query_size, 1).view(-1)
        loss = F.cross_entropy(logits, query_labels)
        accuracy = float(logits.max(-1)[1].eq(query_labels).float().mean())

        return loss, accuracy

    def train_on_loader(self, data_loader, max_iter=None, debug_plot_path=None):
        """Iterate over the training set

        Args:
            data_loader: iterable training data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.train()
        train_loss_meter = BasicMeter.get("train_loss").reset()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(data_loader):
            with torch.autograd.set_detect_anomaly(True):
                loss = self.train_on_batch(batch) / self.exp_dict["tasks_per_batch"]
                train_loss_meter.update(float(loss), 1)
                loss.backward()
                if ((batch_idx + 1) % self.exp_dict["tasks_per_batch"]) == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if batch_idx + 1 == max_iter:
                    break
        return {"train_loss": train_loss_meter.mean()}


    def train_no_episodic_on_loader(self, data_loader, data_un_loader, max_iter=None, debug_plot_path=None):
        """Iterate over the training set

        Args:
            data_loader (torch.utils.data.DataLoader): a pytorch dataloader
            max_iter (int, optional): Max number of iterations if the end of the dataset is not reached. Defaults to None.

        Returns:
            metrics: dictionary with metrics of the training set
        """
        self.model.train()
        train_loss_meter = BasicMeter.get("pre_train_loss").reset()
        # # semco
        # self.embs_dict, self.class_names = data_loader.dataset.embs_dict, data_loader.dataset.class_names
        # self.label_emb_guessor, self.emb_dim = self._get_label_guessor()

        # semco
        # self.ema.update_buffer()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        dl_x, dl_u = iter(data_loader), iter(data_un_loader)
        n_inters = data_loader.dataset.size // data_loader.batch_size
        # freeze model
        if self.exp_dict["freeze"]:
            self.freeze_model()
        # for batch_idx, batch in enumerate(data_loader):
        for i in range(n_inters):
            self.optimizer_freeze.zero_grad()
            batch, batch_un = next(dl_x), next(dl_u)
            loss = self.train_no_episodic_on_batch(batch, batch_un)
            train_loss_meter.update(float(loss), 1)
            loss.backward()
            self.optimizer_freeze.step()
            # semco
            # self.ema.update_params()
            # if batch_idx + 1 == max_iter:
            #     break
        # semco
        # self.ema.update_buffer()
        # self.ema.apply_shadow()
        if self.exp_dict["freeze"]:
            self.unfreeze_model()

        return {"pre_train_loss": train_loss_meter.mean()}

    def generate_label_on_batch(self, batch):
        """Generate the pseudo labels and compute the accuracy on an unlabeled batch

                Args:
                    batch : x (features), y (labels), r (rotation labels), emb (word embedding)

                Returns:
                    tuple: loss and accuracy of the episode
                """
        x, y, r, ims = batch
        y = y[:,0]
        y = y.cuda(non_blocking=True).view(-1)
        # r = r.cuda(non_blocking=True).view(-1)
        k, n, c, h, w = x.size()
        x = x[:,0, ...].view(k, c, h, w).cuda(non_blocking=True)
        if self.ngpu > 1:
            # embeddings = self.parallel_model(x, is_support=True)
            embeddings = self.parallel_model(x)
        else:
            embeddings = self.model(x, is_support=True)
        # if self.exp_dict["embedding_prop"] == True:
        #     embeddings = self.embedding_propagation(embeddings)
        if "embedding_prop_clc" in self.exp_dict.keys():
            if self.exp_dict["embedding_prop_clc"] == True:
                embeddings = self.embedding_propagation(embeddings)

        logits = self.model.classifier(embeddings)
        # loss = F.cross_entropy(logits, y)
        correct_ind = logits.max(-1)[1].eq(y).float()
        acc_all = float(correct_ind.sum())
        # scores = torch.softmax(logits, dim=1)
        # top1, top5 = top_accuracy(scores, y, (1, 5))
        probs = torch.softmax(logits, dim=1)
        scores, lbs_u_guess = torch.max(probs, dim=1)
        logits_max, _ = torch.max(logits, dim=1)
        thr = 0 if "pseudo_thr_g" not in self.exp_dict.keys() else self.exp_dict["pseudo_thr_g"]
        mask = scores.ge(thr)
        ratio = mask.sum()
        if ratio > 0:
            acc_used = (correct_ind * mask).sum()/ ratio
            feature = ims[mask]
            label = lbs_u_guess[mask].cpu()
            label_weight = scores.cpu()
            y = y[mask].cpu()
            logits_max = logits_max[mask].cpu()
            # label = y[mask].cpu()
        else:
            feature, label, label_weight, y, logits_max = [], [], [], [], []
            acc_used = 0

        return feature, label, ratio, acc_used, acc_all, label_weight, logits_max, y

    @torch.no_grad()
    def generate_label_on_loader(self, data_un_loader, max_iter=None, debug_plot_path=None):
        """Iterate over the unlabeled set

        Args:
            data_un_loader (torch.utils.data.DataLoader): a pytorch dataloader
            max_iter (int, optional): Max number of iterations if the end of the dataset is not reached. Defaults to None.

        Returns:
            metrics: dictionary with metrics of the training set
        """
        self.model.eval()
        ratio_meter = BasicMeter.get("used_data_ratio").reset()
        acc_used_meter = BasicMeter.get("accuray_used_data").reset()
        acc_all_meter = BasicMeter.get("accuray_all_data").reset()

        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        dl_u = iter(data_un_loader)
        bs = data_un_loader.batch_size
        n_inters = data_un_loader.dataset.size // bs
        features, labels, labels_weight = [], [], []
        for i in range(n_inters):
            # self.optimizer.zero_grad()
            batch_un = next(dl_u)
            feature, label, ratio, acc_used, acc_all, label_weight, logits_max, y \
                = self.generate_label_on_batch(batch_un)
            ratio_meter.update(float(ratio),1)
            acc_used_meter.update(float(acc_used), 1)
            acc_all_meter.update(float(acc_all), 1)
            if ratio > 0:
                features.append(feature)
                labels.append(label)
                labels_weight.append(label_weight)
        features = torch.cat(features)
        labels = torch.cat(labels)
        labels_weight = torch.cat(labels_weight)
        features = features.numpy().astype('uint8')
        labels = labels.view(-1).numpy()

        return features, labels, labels_weight, {"used_data_ratio": ratio_meter.mean()/bs, "accuracy_used_data": acc_used_meter.mean(),
                                  "accuracy_all_data": acc_all_meter.mean()/bs}

    @torch.no_grad()
    def train_test_on_loader(self, data_loader, max_iter=None, set_name='train_test'):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()
        # # semco
        # self.ema.apply_shadow()
        # self.ema.model.eval()
        ##
        val_loss_meter = BasicMeter.get("val_loss").reset()
        val_accuracy_meter = BasicMeter.get("val_accuracy").reset()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        dl_x = iter(data_loader)
        n_inters = data_loader.dataset.size // data_loader.batch_size
        # next(dl_x)
        # for batch_idx, batch in enumerate(data_loader):
        for i in range(n_inters):
            # for batch_idx, _data in enumerate(data_loader):
            batch = next(dl_x)
            loss, accuracy = self.train_test_on_batch(batch)
            val_loss_meter.update(float(loss), 1)
            val_accuracy_meter.update(float(accuracy), 1)
        # loss = BasicMeter.get(self.exp_dict["target_loss"], recursive=True, force=False).mean()
        # self.scheduler.step(loss)  # update the learning rate monitor
        # # semco
        # # note roll back model current params to continue training
        # self.ema.restore()
        return {"{}_loss".format(set_name): val_loss_meter.mean(),
                "{}_accuracy".format(set_name): val_accuracy_meter.mean()}

    @torch.no_grad()
    def val_on_loader(self, data_loader, max_iter=None):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()
        val_loss_meter = BasicMeter.get("val_loss").reset()
        val_accuracy_meter = BasicMeter.get("val_accuracy").reset()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, _data in enumerate(data_loader):
            batch = _data[0]
            loss, accuracy = self.val_on_batch(batch)
            val_loss_meter.update(float(loss), 1)
            val_accuracy_meter.update(float(accuracy), 1)
        loss = BasicMeter.get(self.exp_dict["target_loss"], recursive=True, force=False).mean()
        self.scheduler.step(loss)  # update the learning rate monitor
        self.scheduler_freeze.step(loss)
        return {"val_loss": val_loss_meter.mean(), "val_accuracy": val_accuracy_meter.mean()}

    @torch.no_grad()
    def test_on_loader(self, data_loader, max_iter=None):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()
        test_loss_meter = BasicMeter.get("test_loss").reset()
        test_accuracy_meter = BasicMeter.get("test_accuracy").reset()
        test_accuracy = []
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, _data in enumerate(data_loader):
            batch = _data[0]
            loss, accuracy = self.val_on_batch(batch)
            test_loss_meter.update(float(loss), 1)
            test_accuracy_meter.update(float(accuracy), 1)
            test_accuracy.append(float(accuracy))
        from scipy.stats import sem, t
        confidence = 0.95
        n = len(test_accuracy)
        std_err = sem(np.array(test_accuracy))
        h = std_err * t.ppf((1 + confidence) / 2, n - 1)
        return {"test_loss": test_loss_meter.mean(), "test_accuracy": test_accuracy_meter.mean(), "test_confidence": h}

    def get_state_dict(self):
        """Obtains the state dict of this model including optimizer, scheduler, etc
        
        Returns:
            dict: state dict
        """
        ret = {}
        ret["optimizer"] = self.optimizer.state_dict()
        ret["model"] = self.model.state_dict()
        ret["scheduler"] = self.scheduler.state_dict()

        ret["optimizer_freeze"] = self.optimizer_freeze.state_dict()
        ret["scheduler_freeze"] = self.scheduler_freeze.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        """Loads the state of the model
        
        Args:
            state_dict (dict): The state to load
        """
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.model.load_state_dict(state_dict["model"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.optimizer_freeze.load_state_dict(state_dict["optimizer_freeze"])
        self.scheduler_freeze.load_state_dict(state_dict["scheduler_freeze"])

    def get_lr(self):
        ret = {}
        for i, param_group in enumerate(self.optimizer.param_groups):
            ret["current_lr_%d" % i] = float(param_group["lr"])
        return ret

    def is_end_of_training(self):
        lr = self.get_lr()["current_lr_0"]
        return lr <= (self.exp_dict["lr"] * self.exp_dict["min_lr_decay"])
