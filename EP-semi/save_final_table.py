import pickle as pkl
import json
import os
import csv
import numpy as np
import pandas as pd
import argparse

fold_name = 'ad739d780d8f2e6f0b825cefee967044'
def load_score(score_dict, dict_tmp, score_name, way_shot):
    n = score_dict.shape[0]

    for i in range(n):
        b = score_dict['backbone'][i]
        if b not in dict_tmp.keys():
            dict_tmp[b] = {}
        num_per_class = score_dict['num_per_class'][i]
        if num_per_class == -1:
            num_per_class = 1000
        npc = '{}_{}_{:03d}'.format(way_shot, score_name, num_per_class)

        ss = {npc: score_dict[score_name][i]}
        dict_tmp[b].update(ss)
        # t = score_dict['test_accuracy'][i]
        # tt = score_dict['train_test_accuracy'][i]
    return dict_tmp

def load_score_list(save_path, pretrain, model_name):
    score_csv = 'pretrain_{}_5_way_1_shot_test_score.csv'.format(pretrain)
    score_dict = pd.read_csv(os.path.join(save_path, score_csv))

    print(score_dict)
    dict_tmp = {}
    dict_tmp = load_score(score_dict, dict_tmp, 'test_accuracy', '5_1')
    score_csv = 'pretrain_{}_5_way_5_shot_test_score.csv'.format(pretrain)
    score_dict = pd.read_csv(os.path.join(save_path, score_csv))
    dict_tmp = load_score(score_dict, dict_tmp, 'test_accuracy', '5_5')
    dict_tmp = load_score(score_dict, dict_tmp, 'train_test_accuracy', '5_5')
    all_score_list = []
    for k, v in dict_tmp.items():
        dict_final = {'model_name': model_name,
                      'training': 'Pretrain' if pretrain else 'Finetune'}
        dict_final.update({'backbone': k})
        v = [(k, v[k]) for k in sorted(v.keys())]
        dict_final.update(v)
        all_score_list += [dict_final]
    return all_score_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sb', '--savedir_base', default='./logs_semi_base_semco_v3/finetune_final_lr001')
    # parser.add_argument('-en', '--exp_name', default='finetune_final_lr001')
    # parser.add_argument('-p', '--pretrain', type=bool, default=False)
    parser.add_argument('-m', '--model_name', default='semco')

    args = parser.parse_args()

    # save_path = os.path.join(args.savedir_base, args.exp_name)
    save_path = args.savedir_base
    model_name = args.model_name
    # pretrain = args.pretrain
    # score_csv = 'pretrain_{}_5_way_1_shot_test_score.csv'.format(pretrain)
    # score_dict = pd.read_csv(os.path.join(save_path, score_csv))
    #
    # print(score_dict)
    # dict_tmp = {}
    # dict_tmp = load_score(score_dict, dict_tmp, 'test_accuracy', '5_1')
    # score_csv = 'pretrain_{}_5_way_5_shot_test_score.csv'.format(pretrain)
    # score_dict = pd.read_csv(os.path.join(save_path, score_csv))
    # dict_tmp = load_score(score_dict, dict_tmp, 'test_accuracy', '5_5')
    # dict_tmp = load_score(score_dict, dict_tmp, 'train_test_accuracy', '5_5')
    # all_score_list = []
    # for k, v in dict_tmp.items():
    #     dict_final = {'model_name':model_name,
    #                   'training':'Pretrain' if pretrain else 'Finetune'}
    #     dict_final.update({'backbone':k})
    #     v = [(k, v[k]) for k in sorted(v.keys())]
    #     dict_final.update(v)
    #     all_score_list += [dict_final]
    # print(all_score_list)
    all_score_list = load_score_list(save_path, True, model_name)
    all_score_list += load_score_list(save_path, False, model_name)
    score_df = pd.DataFrame(all_score_list)
    score_df.to_csv(
        os.path.join(save_path, 'Final_score.csv'))
    print(score_df.tail())



