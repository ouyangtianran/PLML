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

from src import datasets_semi as datasets_emb
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
# pd.set_option("display.max_rows", None, "display.max_columns", None)
import haven
from src.models_semi.utils import remove_prefix

from tensorboardX import SummaryWriter

def load_static_dict(model, exp_dict):
    pretrain_models_flag = 'saved_models'
    if pretrain_models_flag in exp_dict["pretrained_weights_root"]:
        pm = exp_dict["pretrain_method"]
        model_name = exp_dict['model']["backbone"]
        dataset_name = exp_dict['dataset_train_root']
        npc = exp_dict['num_per_class']
        pth_name = f'{pm}_{model_name}_{dataset_name}_npc{npc}'
        for model_pth in os.listdir(exp_dict['pretrained_weights_root']):
            if pth_name in model_pth:
                base_path = os.path.join(exp_dict['pretrained_weights_root'], model_pth)
                model_static_dict = torch.load(base_path)['model_state_dict']
                model_static_dict = remove_prefix(model_static_dict)
                model.load_state_dict(model_static_dict,
                                           strict=False)
                print('load model: ' + model_pth)
                best_accuracy = 0.2
                break
        assert (best_accuracy > 0.1)
    else:
        for exp_hash in os.listdir(exp_dict['pretrained_weights_root']):
            base_path = os.path.join(exp_dict['pretrained_weights_root'], exp_hash)
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
                    model.load_state_dict(torch.load(os.path.join(base_path, 'checkpoint_best.pth'))['model'],
                                               strict=False)
                    if accuracy > best_accuracy:
                        best_path = os.path.join(base_path, 'checkpoint_best.pth')
                        best_accuracy = accuracy
                except:
                    continue
        assert (best_accuracy > 0.1)
        print("Finetuning %s with original accuracy : %f" % (best_path, best_accuracy))
        model.load_state_dict(torch.load(best_path)['model'], strict=False)

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
    # exp_dict["test_iters"] = 10000
    # exp_dict["embedding_prop"] = False
    print('Testing in %s' % savedir)
    test_set = datasets_emb.get_dataset(dataset_name=exp_dict["dataset_test"],
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
    # create model and trainer
    # ==========================
    model = models.get_model(model_name=exp_dict["model"]['name'], backbone=backbone,
                             n_classes=exp_dict["n_classes"],
                             exp_dict=exp_dict,
                             # pretrained_embs=test_set.embs,
                             pretrained_weights_dir=None,
                             savedir_base=savedir_base)
    # model.cpu()
    if not pretrained_weights_dir:
        model.model.load_state_dict(torch.load(os.path.join(savedir, 'checkpoint_best.pth'))['model'],
                              strict=False)
    # else:
    #     load_static_dict(model.model, exp_dict)
    # model.cuda()

    score_list = []
    score_list_path = os.path.join(savedir, f'pretrain_{pretrained_weights_dir}_test_score_list.pkl')
    if not re_test:
        if os.path.exists(score_list_path):
            score_list = hu.load_pkl(score_list_path)

    for way, shot in ways_shots:
        if not re_test:
            break_flag=False
            for sl in score_list:
                if "{} ways {} shots".format(way, shot) == sl["exp_name"]:
                    break_flag=True
            if break_flag:
                continue
        exp_dict["classes_test"] = way
        exp_dict["support_size_test"] = shot
    # load datasets
        test_set = datasets_emb.get_dataset(dataset_name=exp_dict["dataset_test"],
                                            data_root=os.path.join(datadir, exp_dict["dataset_test_root"]),
                                            split="test",
                                            transform=exp_dict["transform_val"],
                                            classes=exp_dict["classes_test"],
                                            support_size=exp_dict["support_size_test"],
                                            query_size=exp_dict["query_size_test"],
                                            n_iters=exp_dict["test_iters"],
                                            unlabeled_size=exp_dict["unlabeled_size_test"])
        # split_train = 'train-test-seed-10-K-{:03d}'.format(exp_dict["num_per_class"])
        if exp_dict["num_per_class"] > 0:
            split_train = 'train-test-seed-10-K-{:03d}'.format(exp_dict["num_per_class"])
            data_group_train = 'train_label'
        elif exp_dict["num_per_class"] == -1:
            split_train = 'train-test-seed-10-K-{:03d}'.format(20)
            data_group_train = 'train_all'
        train_test_set = datasets_emb.get_dataset(dataset_name=exp_dict["dataset_train_test"],
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
        # get dataloaders
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x,
            drop_last=True)
        train_test_loader = torch.utils.data.DataLoader(
            train_test_set,
            batch_size=exp_dict["semi_batch_size"],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=ut.get_collate("default"),
            drop_last=True)

        score_dict = {"exp_name": "{} ways {} shots".format(way, shot)}
        test_dict = model.test_on_loader(test_loader, max_iter=None)
        score_dict.update(test_dict)
        # train_test
        score_dict.update(model.train_test_on_loader(train_test_loader, set_name='train_test'))
        score_list += [score_dict]
        # Report
        score_df = pd.DataFrame(score_list)
        print(score_df.tail())
    hu.save_pkl(score_list_path, score_list)
    score_df = pd.DataFrame(score_list)
    score_df.to_csv(os.path.join(savedir, f'pretrain_{pretrained_weights_dir}_test_score.csv'))

    return score_list




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', default='./logs_semi_base_semco_v3/finetune_final')
    parser.add_argument('-d', '--datadir', default='./data')
    parser.add_argument('-r', '--reset', default=0, type=int)
    parser.add_argument('-ei', '--exp_id', type=str, default=None)
    parser.add_argument('-j', '--run_jobs', type=int, default=0)
    parser.add_argument('-nw', '--num_workers', default=0, type=int)
    # parser.add_argument('-p', '--pretrained_weights_dir', type=str, default=None)
    parser.add_argument('-p', '--pretrain', type=bool, default=False)

    args = parser.parse_args()

    ways_shots = [[5, 1], [5, 5]]
    out_parameters = ["num_per_class", "model.backbone", "classification_weight"]  # , "resort", "query_drop_rate"
    ll = os.listdir(args.savedir_base)
    for ws0 in ways_shots:
        way, shot = ws0
        all_score_list = []
        for l in ll:
            if '.' in l:
                continue
            # if not l == '72b79334bb86f7b5c62ff47c78726705':
            #     continue
            print(l)
            exp_dict = hu.load_json(os.path.join(args.savedir_base, l, 'exp_dict.json'))

            score_list = testing(exp_dict, args.savedir_base, args.datadir, [ws0], exp_id=l,
                                 pretrained_weights_dir=args.pretrain, re_test=args.reset)
            names = exp_dict.keys()
            for sl in score_list:
                if sl['exp_name'] == "{} ways {} shots".format(way, shot):
                    for op in out_parameters:
                        op_split = op.split('.')
                        if len(op_split) > 1:
                            js_tmp = exp_dict
                            for op0 in op_split:
                                out0 = js_tmp[op0]
                                js_tmp = out0
                            # print_str += '{}: {}, '.format(op0, out0)
                            s_dict = {op0: out0}
                            sl.update(s_dict)
                        elif op in names:
                            s_dict = {op:exp_dict[op]}
                            sl.update(s_dict)
                    all_score_list += [sl]

        score_df = pd.DataFrame(all_score_list)
        score_df.to_csv(os.path.join(args.savedir_base, 'pretrain_{}_{}_way_{}_shot_test_score.csv'.format(args.pretrain, way, shot)))
        print(score_df.tail())
