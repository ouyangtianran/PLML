import pickle as pkl
import json
import os
import csv
import numpy as np
import pandas as pd
import argparse

fold_name = 'ad739d780d8f2e6f0b825cefee967044'
def load_score(score_dict, dict_tmp, score_name, way_shot='5_1'):
    n = score_dict.shape[0]

    for i in range(n):
        b = score_dict['backbone'][i]
        if b not in dict_tmp.keys():
            dict_tmp[b] = {}
        num_per_class = score_dict['num_per_class'][i]
        shot = score_dict['support_size_test'][i]
        if num_per_class == -1:
            num_per_class = 1000
        npc = '{}_{}_{:03d}'.format(shot, score_name, num_per_class)

        ss = {npc: score_dict[score_name][i]}
        dict_tmp[b].update(ss)
        # t = score_dict['test_accuracy'][i]
        # tt = score_dict['train_test_accuracy'][i]
    return dict_tmp

def load_score_list(score_df, score_names, model_name):

    dict_tmp = {}
    for sn in score_names:
        dict_tmp = load_score(score_df, dict_tmp, sn)

    all_score_list = []
    for k, v in dict_tmp.items():
        dict_final = {'model_name': model_name}
        dict_final.update({'backbone': k})
        v = [(k, v[k]) for k in sorted(v.keys())]
        dict_final.update(v)
        all_score_list += [dict_final]
    return all_score_list

def print_score(root, fold_name, compare_papameters, score_names):
    file_name = '{}/{}/score_list.pkl'.format(root, fold_name)
    json_name = '{}/{}/exp_dict.json'.format(root, fold_name)
    out_dict = {}
    with open(json_name) as f:
        js = json.load(f)
    names = js.keys()
    print_names = compare_papameters
        #['num_per_class',"unsupervised_loss","unlabeled_multiple","rotation_weight","pseudo_thr",
                   # "lr", "semi_batch_size"]
    print_str = ""

    # shot_name = ''
    for pn in print_names:
        pn_split = pn.split('.')
        if len(pn_split)>1:
            js_tmp = js
            for pn0 in pn_split:
                out0 = js_tmp[pn0]
                js_tmp = out0
            print_str += '{}: {}, '.format(pn0, out0)
            out_dict[pn0] = out0
            # out_list.append('{}{}'.format(out0,shot_name))
        # else:
        elif pn in names:
            print_str += '{}: {}, '.format(pn, js[pn])
            out0 = js[pn]
            out_dict[pn] = out0
            # out_list.append('{}{}'.format(out0,shot_name))

    with open(file_name, 'rb') as infile:
        data = pkl.load(infile)

    acc_max = 0
    print_names_acc = score_names#['epoch', 'test_accuracy', 'test_confidence', 'val_accuracy', 'train_test_accuracy']
    ur={"used_data_ratio":0, "accuracy_used_data":0}
    for d in data:
        val_acc = d['ssl_accuracy']
        if "used_data_ratio" in d.keys():
            ur["used_data_ratio"] = d["used_data_ratio"]
            ur["accuracy_used_data"] = d["accuracy_used_data"]
        if val_acc > acc_max:
            # ss = 'epcoh: {}, val_accuracy: {},test_accuracy: {}, test_confidence: {}'.format(e, val_acc, acc, con)
            d.update(ur)
            names = d.keys()
            print_str_acc = ""
            s_dict ={}
            for pn in print_names_acc:
                if pn in names:
                    print_str_acc += '{}: {}, '.format(pn, d[pn])
                    # s_list.append(d[pn])
                    s_dict[pn]=d[pn]
            ss = print_str_acc
            acc_max = val_acc

    if 1:#js["lr"] == 0.2:
        print(fold_name)
        print(print_str)
        print('max: {}'.format(ss))
    # out_list = out_list + s_list
    out_dict.update(s_dict)
    return out_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sb', '--savedir_base', default='./logs_semi_base_semco_v3/ssl')
    # parser.add_argument('-en', '--exp_name', default='finetune_final_lr001')
    # parser.add_argument('-p', '--pretrain', type=bool, default=False)
    parser.add_argument('-m', '--model_name', default='semco')

    args = parser.parse_args()


    # save_path = os.path.join(args.savedir_base, args.exp_name)
    save_path = args.savedir_base
    model_name = args.model_name
    compare_papameters = ["model.backbone", "num_per_class","support_size_test", "n_distract"]
    score_names = ['ssl_accuracy']
    ll = os.listdir(save_path)
    outs = []
    print(save_path)
    for l in ll:
        if '.' in l:
            continue
        out_list = print_score(save_path, l, compare_papameters, score_names)
        outs.append(out_list)

    score_df = pd.DataFrame(outs)
    all_score_list = load_score_list(score_df, score_names, model_name)

    score_df = pd.DataFrame(all_score_list)
    score_df.to_csv(
        os.path.join(save_path, 'Final_ssl_score.csv'))
    print(score_df.tail())



