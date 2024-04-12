import torch
import argparse
import pandas as pd
import sys
import os
from torch import nn
from torch.nn import functional as F
import tqdm
import pprint
from src import utils as ut
import torchvision
import numpy as np

from src import datasets_emb_semi as datasets_emb
from src import models_emb_semi as models
from src.models_emb_semi import backbones
from torch.utils.data import DataLoader
import exp_configs_emb_semi as exp_configs
from torch.utils.data.sampler import RandomSampler

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jupyter as hj
from sklearn import manifold
from sklearn.preprocessing import normalize, StandardScaler
# pd.set_option("display.max_rows", None, "display.max_columns", None)
import haven
from src.models_emb_semi.utils import top_accuracy, remove_prefix
from src.tools.meters import BasicMeter
import matplotlib.pyplot as plt

@torch.no_grad()
def get_loss(model, data_un_loader):
    model.model.eval()
    model.exp_dict["pseudo_thr_g"] = 0
    ratio_meter = BasicMeter.get("used_data_ratio").reset()
    acc_used_meter = BasicMeter.get("accuray_used_data").reset()
    acc_all_meter = BasicMeter.get("accuray_all_data").reset()

    # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
    dl_u = iter(data_un_loader)
    bs = data_un_loader.batch_size
    n_inters = data_un_loader.dataset.size // bs
    features, labels, labels_weight, gt, lm = [], [], [], [], []
    for i in range(n_inters):
        # model.optimizer.zero_grad()
        batch_un = next(dl_u)
        feature, label, ratio, acc_used, acc_all, label_weight, logits_max, y \
            = model.generate_label_on_batch(batch_un)
        ratio_meter.update(float(ratio), 1)
        acc_used_meter.update(float(acc_used), 1)
        acc_all_meter.update(float(acc_all), 1)
        if ratio > 0:
            features.append(feature)
            labels.append(label)
            labels_weight.append(label_weight)
            gt.append(y)
            lm.append(logits_max)
    features = torch.cat(features)
    labels = torch.cat(labels)
    labels_weight = torch.cat(labels_weight)
    lm = torch.cat(lm)
    gt = torch.cat(gt)
    p = labels_weight
    loss = -p*p.log()
    mask = labels.eq(gt)
    features = features.numpy().astype('uint8')
    labels = labels.view(-1).numpy()
    return loss.numpy(), mask.numpy(), p.numpy(), lm.numpy()

def plot_hist(loss, mask, hist_path):
    l_correct = loss[mask]
    l_mislabel = loss[mask==0]
    plt.hist([l_correct, l_mislabel], label=['correct', 'mislabel'])
    plt.legend(loc='upper left')
    # plt.show()
    plt.savefig(hist_path)
    plt.close()

def testing(exp_dict, savedir_base, datadir, ways_shots, exp_id,
             num_workers=0, pretrained_weights_dir=False, re_test=False):
    # bookkeeping
    # ---------------

    # get experiment directory
    # exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)
    if not pretrained_weights_dir:
        exp_dict["pretrained_weights_root"] = None
    exp_dict["ngpu"] = 1
    exp_dict["batch_size"] = 128
    # exp_dict["embedding_prop"] = False
    print('Testing in %s' % savedir)
    score_list_path = os.path.join(savedir, f'pretrain_{pretrained_weights_dir}_loss_list.pkl')
    if not re_test and os.path.exists(score_list_path):
        score_list = hu.load_pkl(score_list_path)
        loss, mask, p, logits_max = score_list
    else:
        if exp_dict["num_per_class"] > 0:
            split_train = 'train-test-seed-10-K-{:03d}'.format(exp_dict["num_per_class"])
        train_un_set = datasets_emb.get_dataset(dataset_name=exp_dict["dataset_train_test"],
                                                data_root=os.path.join(datadir, exp_dict["dataset_train_root"]),
                                                split=split_train,
                                                transform=exp_dict["transform_train"],
                                                classes=exp_dict["classes_train"],
                                                support_size=exp_dict["support_size_train"],
                                                query_size=exp_dict["query_size_train"],
                                                n_iters=exp_dict["train_iters"],
                                                unlabeled_size=exp_dict["unlabeled_size_train"],
                                                rotation_labels=exp_dict["rotation_labels"],
                                                unsupervised_loss=exp_dict["unsupervised_loss"],
                                                data_group='train_unlabel')
        train_un_loader = torch.utils.data.DataLoader(
            train_un_set,
            batch_size=64,#128,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=ut.get_collate("default"),
            drop_last=True)

        # Create model, opt, wrapper
        backbone = backbones.get_backbone(backbone_name=exp_dict['model']["backbone"], exp_dict=exp_dict)
        # create model and trainer
        # ==========================
        model = models.get_model(model_name=exp_dict["model"]['name'], backbone=backbone,
                                 n_classes=exp_dict["n_classes"],
                                 exp_dict=exp_dict,
                                 pretrained_embs=train_un_set.embs,
                                 pretrained_weights_dir=None,
                                 savedir_base=savedir_base)
        # model.cpu()
        if not pretrained_weights_dir:
            model.model.load_state_dict(torch.load(os.path.join(savedir, 'checkpoint_best.pth'))['model'],
                                  strict=False)
        # else:
        #     load_static_dict(model.model, exp_dict)
        # model.cuda()
        # score_list = []

        loss, mask, p, logits_max = get_loss(model, train_un_loader)
        score_list = [loss, mask, p, logits_max]
        hu.save_pkl(score_list_path, score_list)
    hist_path = os.path.join(savedir, f'pretrain_{pretrained_weights_dir}_loss_hist.png')
    plot_hist(np.log(loss+1e-6), mask, hist_path)
    hist_path = os.path.join(savedir, f'pretrain_{pretrained_weights_dir}_prob_hist.png')
    # plot_hist(np.log(p + 1e-6), mask, hist_path)
    plot_hist(p, mask, hist_path)
    hist_path = os.path.join(savedir, f'pretrain_{pretrained_weights_dir}_p_logits_hist.png')
    # plot_hist(np.log(p + 1e-6), mask, hist_path)
    loss2 = -p * np.log(logits_max)
    plot_hist(loss2, mask, hist_path)
    # score_list = []
    # score_list_path = os.path.join(savedir, f'pretrain_{pretrained_weights_dir}_acc_generate_list.pkl')
    # if not re_test:
    #     if os.path.exists(score_list_path):
    #         score_list = hu.load_pkl(score_list_path)
    # score_dict = {}#{"delay_iter":exp_dict["delay_iter"], "pseudo_thr_g": exp_dict["pseudo_thr_g"]}
    # out = model.generate_label_on_loader(train_un_loader, max_iter=None)
    # test_dict = out[2]
    # score_dict.update(test_dict)
    # # # train_test
    # # score_dict.update(model.train_test_on_loader(train_test_loader, set_name='train_test'))
    # score_list += [score_dict]
    # # Report
    # score_df = pd.DataFrame(score_list)
    # print(score_df.tail())
    #
    # hu.save_pkl(score_list_path, score_list)
    # score_df = pd.DataFrame(score_list)
    # score_df.to_csv(os.path.join(savedir, f'pretrain_{pretrained_weights_dir}_test_score.csv'))
    #
    # return score_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', default='./logs_semi_base_semco_v3/finetune_delay2')
    parser.add_argument('-d', '--datadir', default='./data')
    parser.add_argument('-r', '--reset', default=0, type=int)
    parser.add_argument('-ei', '--exp_id', type=str, default=None)
    parser.add_argument('-j', '--run_jobs', type=int, default=0)
    parser.add_argument('-nw', '--num_workers', default=0, type=int)
    # parser.add_argument('-p', '--pretrained_weights_dir', type=str, default=None)
    parser.add_argument('-p', '--pretrain', type=bool, default=True)
    args = parser.parse_args()

    out_parameters = ["delay_iter", "pseudo_thr_g"]  # "classification_weight"
    all_score_list = []
    ll = os.listdir(args.savedir_base)
    # for ws0 in ways_shots:
    #     way, shot = ws0
    #     all_score_list = []
    ll = ['9ca8b4ef167de17b0a4e96f86fb009e0']
    for l in ll:
        if '.' in l:
            continue
        # l = '9ca8b4ef167de17b0a4e96f86fb009e0'
        exp_dict = hu.load_json(os.path.join(args.savedir_base, l, 'exp_dict.json'))
        testing(exp_dict, args.savedir_base, args.datadir, [[5, 1]], exp_id=l,
                pretrained_weights_dir=args.pretrain, re_test=args.reset)

    #     score_list = testing(exp_dict, args.savedir_base, args.datadir, [[5,1]], exp_id=l,
    #                                      pretrained_weights_dir=args.pretrain, re_test=True)
    #     names = exp_dict.keys()
    #     for sl in score_list:
    #         # if sl['exp_name'] == "{} ways {} shots".format(way, shot):
    #             for op in out_parameters:
    #                 if op in names:
    #                     s_dict = {op: exp_dict[op]}
    #                     sl.update(s_dict)
    #             all_score_list += [sl]
    # score_df = pd.DataFrame(all_score_list)
    # score_df.to_csv(
    #     os.path.join(args.savedir_base, 'pretrain_{}_acc_generate.csv'.format(args.pretrain)))
    # print(score_df.tail())