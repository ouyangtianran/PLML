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

from src import datasets_semi as datasets
from src import models_semi as models
from src.models_semi import backbones
from torch.utils.data import DataLoader
import exp_configs_emb_semi as exp_configs
from torch.utils.data.sampler import RandomSampler

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jupyter as hj
from sklearn import manifold
from sklearn.preprocessing import normalize, StandardScaler
pd.set_option("display.max_rows", None, "display.max_columns", None)

from tensorboardX import SummaryWriter

def trainval(exp_dict, savedir_base, datadir, reset=False,
             num_workers=0, pretrained_weights_dir=None):
    # bookkeeping
    # ---------------

    # get experiment directory
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)

    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    pprint.pprint(exp_dict)
    print('Experiment saved in %s' % savedir)

    # load datasets
    # ==========================
    if exp_dict["num_per_class"] > 0:
        split_train = 'train-test-seed-10-K-{:03d}'.format(exp_dict["num_per_class"])
        data_group_train = 'train_label'
    elif exp_dict["num_per_class"] == -1:
        split_train = 'train-test-seed-10-K-{:03d}'.format(20)
        data_group_train = 'train_all'
    aug = False
    if "cluster-FSL" in exp_dict.keys():
        if exp_dict["cluster-FSL"]==True:
            aug=True
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset_train"],
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
                                         data_group=data_group_train,
                                         aug=aug)
# train_set0 和 train_un_set 是用于semi-EP的
    train_set0 = datasets.get_dataset(dataset_name=exp_dict["dataset_train_test"],
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
                                         data_group=data_group_train)

    train_un_set = datasets.get_dataset(dataset_name=exp_dict["dataset_train_test"],
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
                                         data_group='train_unlabel',
                                        #n_distract=exp_dict["n_distract"]
                                        )

    train_test_set = datasets.get_dataset(dataset_name=exp_dict["dataset_train_test"],
                                         data_root=os.path.join(datadir, exp_dict["dataset_train_root"]),
                                         split=split_train,
                                         transform=exp_dict["transform_train"],
                                         classes=exp_dict["classes_train"],
                                         support_size=exp_dict["support_size_train"],
                                         query_size=exp_dict["query_size_train"],
                                         n_iters=exp_dict["train_iters"],
                                         unlabeled_size=0,
                                         rotation_labels=[0],
                                         unsupervised_loss='',
                                         data_group='train_test')

    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset_val"],
                                       data_root=os.path.join(datadir, exp_dict["dataset_val_root"]),
                                       split="val",
                                       transform=exp_dict["transform_val"],
                                       classes=exp_dict["classes_val"],
                                       support_size=exp_dict["support_size_val"],
                                       query_size=exp_dict["query_size_val"],
                                       n_iters=exp_dict.get("val_iters", None),
                                       unlabeled_size=exp_dict["unlabeled_size_val"])

    test_set = datasets.get_dataset(dataset_name=exp_dict["dataset_test"],
                                        data_root=os.path.join(datadir, exp_dict["dataset_test_root"]),
                                        split="test",
                                        transform=exp_dict["transform_val"],
                                        classes=exp_dict["classes_test"],
                                        support_size=exp_dict["support_size_test"],
                                        query_size=exp_dict["query_size_test"],
                                        n_iters=exp_dict["test_iters"],
                                        unlabeled_size=exp_dict["unlabeled_size_test"])


    # Create model, opt, wrapper
    backbone = backbones.get_backbone(backbone_name=exp_dict['model']["backbone"], exp_dict=exp_dict)


     # get dataloaders
    # ==========================
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=exp_dict["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ut.get_collate(exp_dict["collate_fn"]),
        drop_last=True)
    train_loader0 = torch.utils.data.DataLoader(
        train_set0,
        batch_size=exp_dict["semi_batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ut.get_collate("default"),
        drop_last=True)
    train_un_loader = torch.utils.data.DataLoader(
        train_un_set,
        batch_size=max(exp_dict["semi_batch_size"]* exp_dict["unlabeled_multiple"],64),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ut.get_collate("default"),
        drop_last=True)
    train_test_loader = torch.utils.data.DataLoader(
        train_test_set,
        batch_size=exp_dict["semi_batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ut.get_collate("default"),
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        drop_last=True)

    # create model and trainer
    # ==========================
    model = models.get_model(model_name=exp_dict["model"]['name'], backbone=backbone,
                             n_classes=exp_dict["n_classes"],
                             exp_dict=exp_dict,
                             pretrained_embs=None,
                             pretrained_weights_dir=pretrained_weights_dir,
                             savedir_base=savedir_base)

    # Pretrain or Fine-tune or run SSL
    if exp_dict["model"]['name'] == 'ssl' or exp_dict["model"]['name'] == 'semi_ssl':
        # runs the SSL experiments
        score_list_path = os.path.join(savedir, 'score_list.pkl')
        if not os.path.exists(score_list_path):
            test_dict = model.test_on_loader(test_loader, max_iter=None)
            hu.save_pkl(score_list_path, [test_dict])
        return

        # Checkpoint
    # -----------
    checkpoint_path = os.path.join(savedir, 'checkpoint.pth')
    score_list_path = os.path.join(savedir, 'score_list.pkl')

    if os.path.exists(score_list_path):
        # resume experiment
        model.load_state_dict(hu.torch_load(checkpoint_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0
    # tensorboardX
    writer = SummaryWriter(comment=f'{exp_id}')

    features0 = train_loader.dataset.features
    labels0 = train_loader.dataset.labels
    labels0_weight = train_loader.dataset.labels_weight
    lr0 = 0

    min_npc=20
    if exp_dict["num_per_class"]<min_npc and exp_dict["num_per_class"]>0:
        nn = exp_dict["num_per_class"]
        while nn < min_npc:
            features0 = np.concatenate([features0, features0])
            labels0 = np.concatenate([labels0, labels0])
            labels0_weight = np.concatenate([labels0_weight, labels0_weight])
            nn=nn*2

    def update_train_loader(epoch):
        if epoch==0:
            if "embedding_prop_clc" in model.exp_dict.keys():
                ep0 = model.exp_dict["embedding_prop_clc"]
                model.exp_dict["embedding_prop_clc"] = False

        features, labels, labels_weight, score_meters, gts = model.generate_label_on_loader(train_un_loader)
        print('unlabel data length: {}'.format(len(labels)))

        if epoch==0:
            if "embedding_prop_clc" in model.exp_dict.keys():
                model.exp_dict["embedding_prop_clc"] = ep0
        score_dict.update(score_meters)


        if exp_dict["unlabeled_size_train"] == 0:
            if "only_pseudo_labels" in exp_dict.keys():
                print('only_pseudo_labels')
                labels_weight = labels_weight.numpy()
            else:
                features = np.concatenate([features0, features])
                labels = np.concatenate([labels0, labels])
                labels_weight = np.concatenate([labels0_weight, labels_weight])
                gts = np.concatenate([labels0, gts])
            # update train loader
            train_loader.dataset.features = features
            train_loader.dataset.labels = labels
            train_loader.dataset.labels_gt = gts
            train_loader.dataset.labels_weight = labels_weight
            train_loader.dataset.indices = np.arange(len(labels))
            train_loader.dataset.labelset = np.unique(labels)
            train_loader.dataset.reshuffle()
        elif exp_dict["unlabeled_size_train"] > 0:
            # update train loader
            train_loader.dataset.un_features = features
            train_loader.dataset.un_labels = labels
            train_loader.dataset.un_labels_weight = labels_weight
            train_loader.dataset.un_indices = np.arange(len(labels))
            train_loader.dataset.reshuffle()
    if "pseudo_thr_g" in exp_dict.keys():
        if exp_dict["pseudo_thr_g"] == -1:
            dynamical_ptg = -1
            pseudo_label_num = 5
            pseudo_thr_g0 = 1
        elif exp_dict["pseudo_thr_g"] == -2:
            dynamical_ptg = -2
            pseudo_label_num = 5
            pseudo_thr_g0 = -1.0/pseudo_label_num
        else:
            dynamical_ptg = False
    else:
        dynamical_ptg = False

    # Run training and validation
    for epoch in range(s_epoch, exp_dict["max_epoch"]):
        score_dict = {"epoch": epoch}
        lr_dict = model.get_lr()
        score_dict.update(lr_dict)
        current_lr = lr_dict["current_lr_0"]

        if exp_dict["model"]['name'] == 'semi_pretraining':
            # train
            score_dict.update(model.train_on_loader(train_loader0, train_un_loader))

            # train_test
            score_dict.update(model.train_test_on_loader(train_test_loader, set_name='train_test'))
        elif exp_dict["model"]['name'] == 'semi_finetuning':
            # train
            if exp_dict["use_unlabeled_data"] == "iterating_optimization":
                if exp_dict["delay_iter"] == -1:
                    if current_lr != lr0:
                        lr0 = current_lr
                        if dynamical_ptg==-1:
                            pseudo_thr_g0 = max(pseudo_thr_g0-1.0/pseudo_label_num, 0)
                            model.exp_dict["pseudo_thr_g"] = pseudo_thr_g0
                        elif dynamical_ptg==-2:
                            pseudo_thr_g0 = min(pseudo_thr_g0 + 1.0/pseudo_label_num, 0.95)
                            model.exp_dict["pseudo_thr_g"] = pseudo_thr_g0
                        update_train_loader(epoch)
                elif exp_dict["delay_iter"] == 0:
                    if epoch == s_epoch:
                        update_train_loader(epoch)
                elif epoch % exp_dict["delay_iter"] == 0:
                    update_train_loader(epoch)

                score_dict.update(model.train_no_episodic_on_loader(train_loader0, train_un_loader))
                score_dict.update(model.train_on_loader(train_loader))
            # elif exp_dict["use_unlabeled_data"] == "no_iteration":
            #     if epoch == s_epoch:
            #         update_train_loader(epoch)
            #     score_dict.update(model.train_no_episodic_on_loader(train_loader0, train_un_loader))
            #     score_dict.update(model.train_on_loader(train_loader))
            elif exp_dict["use_unlabeled_data"] == "only_pseudo":
                if epoch == s_epoch:
                    update_train_loader(epoch)
                score_dict.update(model.train_on_loader(train_loader))
            else:
                score_dict.update(model.train_on_loader(train_loader))
            score_dict.update(model.train_test_on_loader(train_test_loader, set_name='train_test'))
        elif exp_dict["model"]['name'] == 'noise_finetuning':
            if epoch == s_epoch and data_group_train != 'train_all':
                update_train_loader(epoch)
            # train_test
            score_dict.update(model.train_test_on_loader(train_test_loader, set_name='train_test'))
            
            score_dict.update(model.train_on_loader(train_loader))

        # elif exp_dict["model"]['name'] == 'semi_ssl':
        #     score_dict.update(model.train_test_on_loader(train_test_loader, set_name='train_test'))
        # else:
        #     raise NotImplementedError

        # validate
        # if exp_dict["model"]['name'] != 'semi_ssl':
        score_dict.update(model.val_on_loader(val_loader))
        score_dict.update(model.test_on_loader(test_loader))

        # write to tensorboardX
        for s_name in score_dict:
            if 'loss' in s_name:
                s_name1 = 'loss/'+s_name
            elif 'accuracy' in s_name:
                s_name1 = 'acc/' + s_name
            else:
                s_name1 = s_name

            writer.add_scalar(s_name1, score_dict[s_name], epoch)

        # Add score_dict to score_list
        score_list += [score_dict]

        # Report
        score_df = pd.DataFrame(score_list)
        print(score_df.tail())

        # Save checkpoint
        hu.save_pkl(score_list_path, score_list)
        hu.torch_save(checkpoint_path, model.get_state_dict())
        print("Saved: %s" % savedir)

        if "accuracy" in exp_dict["target_loss"]:
            is_best = score_dict[exp_dict["target_loss"]] >= score_df[exp_dict["target_loss"]][:-1].max()
        else:
            is_best = score_dict[exp_dict["target_loss"]] <= score_df[exp_dict["target_loss"]][:-1].min()

            # Save best checkpoint
        if is_best:
            hu.save_pkl(os.path.join(savedir, "score_list_best.pkl"), score_list)
            hu.torch_save(os.path.join(savedir, "checkpoint_best.pth"), model.get_state_dict())
            print("Saved Best: %s" % savedir)

            # Check for end of training conditions
        if model.is_end_of_training():
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', default='')
    parser.add_argument('-r', '--reset', default=0, type=int)
    parser.add_argument('-ei', '--exp_id', type=str, default=None)
    parser.add_argument('-j', '--run_jobs', type=int, default=0)
    parser.add_argument('-nw', '--num_workers', default=0, type=int)
    parser.add_argument('-p', '--pretrained_weights_dir', type=str, default=None)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))

        exp_list = [exp_dict]

    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # Run experiments or View them
    # ----------------------------
    if args.run_jobs:
        pass
    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                     savedir_base=args.savedir_base,
                     datadir=args.datadir,
                     reset=args.reset,
                     num_workers=args.num_workers,
                     pretrained_weights_dir=args.pretrained_weights_dir)
