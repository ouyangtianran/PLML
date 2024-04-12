import pickle as pkl
import json
import os
import csv
import numpy as np
import pandas as pd

fold_name = 'ad739d780d8f2e6f0b825cefee967044'


def print_score(root, fold_name, compare_papameters, score_names):
    file_name = '{}/{}/score_list.pkl'.format(root, fold_name)
    json_name = '{}/{}/exp_dict.json'.format(root, fold_name)
    out_list = []
    with open(json_name) as f:
        js = json.load(f)
    names = js.keys()
    print_names = compare_papameters
        #['num_per_class',"unsupervised_loss","unlabeled_multiple","rotation_weight","pseudo_thr",
                   # "lr", "semi_batch_size"]
    print_str = ""
    shot = js["support_size_test"]
    # shot_name = ''
    for pn in print_names:
        if pn == print_names[0]:
            shot_name = '-{}_shot'.format(shot)
        else:
            shot_name = ''
        pn_split = pn.split('.')
        if len(pn_split)>1:
            js_tmp = js
            for pn0 in pn_split:
                out0 = js_tmp[pn0]
                js_tmp = out0
            print_str += '{}: {}, '.format(pn0, out0)
            out_list.append('{}{}'.format(out0,shot_name))
        # else:
        elif pn in names:
            print_str += '{}: {}, '.format(pn, js[pn])
            out0 = js[pn]
            out_list.append('{}{}'.format(out0,shot_name))

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
            s_list =[]
            for pn in print_names_acc:
                if pn in names:
                    print_str_acc += '{}: {}, '.format(pn, d[pn])
                    s_list.append(d[pn])
            ss = print_str_acc
            acc_max = val_acc

    if 1:#js["lr"] == 0.2:
        print(fold_name)
        print(print_str)
        print('max: {}'.format(ss))
    out_list = out_list + s_list
    return out_list

d = './logs_semi_base_semco/pretraining'
d = './logs_semi_base_semco/finetune_dy_pseudo_thr_g'
d = './logs_semi_base_semco_v3/finetune_final_lr001'
# d = './logs_semi_base_semco_ab/finetune_best_pre_freeze_layers2'
# d = '/home/xd1/code/EP-semi/logs_semi_base/finetune'
d = './logs_semi_base_norm/pretrain_cub'
d = './logs_semi_base_norm/finetune_cub_lr0001'
d = './logs_semi_base_norm/finetune_tiered_lr0001'
# d = './logs_semi_base_semco_v3/finetune_tiered'
# d = './logs_semi_base_norm/pretrain_tiered'
# d = './logs_semi_base_norm/pretraining3'
d = './logs_semi_base_norm/finetune3'
d = './logs_semi_base_norm/finetune_tiered_cw01'
d = './logs_semi_base_semco_v3/ssl'
# d = './logs_semi_base_norm/finetune3_cw01'

# d = './logs_semi_base_semco_v3/finetune_hp_thr_g'
# d = './logs_semi_base_semco_v3/finetune_cw01'
# d = './logs_semi_base_semco_v3/finetune_hp3'
# d = './logs_semi_base_norm/pretrain_cub'
# d = './logs_semi_base_semco_v3/finetune_cub'
# d = './logs_semi_base_norm/pretrain_tiered'
# d = './logs_semi_base_norm/finetune_tiered'
# d = './logs_semi_base_semco_ab/finetune_best_pre_cw1'
# # d = './logs_semi_base_norm_cw/finetune'
# # d = './logs_semi_base_norm/finetune_cw15'
# d = './logs_semi_base_semco_low_shot/pretraining'
# d = './logs_semi_base_semco_low_shot/finetune'
# d = './logs_lr001/pretraining'
# d = './logs_lr001/finetune'
# d = './logs_emb_semi/finetune'
# d = '/home/xd1/code/EP-semi/logs_emb_semi/finetune'
# d = '/home/xd1/code/embedding-propagation/logs/pretraining'
# d = '/home/xd1/code/EP-semi/logs/ssl'

compare_papameters =[ "classification_weight","num_per_class"]
# compare_papameters =["use_unlabeled_data", "classification_weight"] #use_unlabeled_data "model.backbone","pseudo_thr_g", "classification_weight", "model.backbone", 'lr','unlabeled_multiple''unlabeled_multiple', 'freeze','classification_weight'  'pseudo_thr',only for two parameters
compare_papameters =[ "model.backbone","num_per_class"] #"unlabeled_size_train" "p_thr_g" "embedding_prop_clc"
# compare_papameters =[ "unlabeled_size_train","unlabeled_multiple"]
# compare_papameters =[ "model.backbone","lr"]
# compare_papameters =["use_unlabeled_data", "pseudo_thr_g"]
# compare_papameters =["delay_iter", "pseudo_thr_g"] #embedding_prop_clc
# compare_papameters =[ "use_unlabeled_data","num_per_class"]#"delay_iter","embedding_prop_clc",
# compare_papameters =["pseudo_thr_g","embedding_prop_clc"]
# compare_papameters =["classification_weight","lr"]

# score_names = ['train_test_accuracy','test_accuracy','val_accuracy']#,'used_data_ratio','accuracy_used_data','val_accuracy'
score_names = ['ssl_accuracy']
ll = os.listdir(d)
outs = []
print(d)
for l in ll:
    if '.' in l:
        continue
    out_list = print_score(d, l, compare_papameters, score_names)
    outs.append(out_list)
a = np.array(outs)
# re-arrage the out array
p0_unique = np.unique(a[:,0])
p1_unique = np.unique(a[:,1])
n0, n1 = len(p0_unique), len(p1_unique)
n_exp = a.shape[0]

n_scores = len(score_names)
for i in range(n_scores):
    a_final = np.empty((n0 + 1, n1 + 1),dtype=object)
    a_tmp = np.zeros((n0, n1))
    csv_name = '{} {} {}.csv'.format(compare_papameters[0], compare_papameters[1], score_names[i])
    save_file = os.path.join(d, csv_name)
    for a0 in a:
        ind0 = np.where(p0_unique==a0[0])
        ind1 = np.where(p1_unique==a0[1])
        a_tmp[ind0[0][0],ind1[0][0]] = a0[i+2]
    a_final[1:,0] = p0_unique
    a_final[0,1:] = p1_unique
    a_final[1:,1:] = a_tmp
    # a_final.tofile(csv_name,sep=',',format='%10.5f')
    # np.savetxt(save_file, a_final, delimiter=',',fmt='%10.5f')
    np.savetxt(save_file, a_final, delimiter=',', fmt='%s')

# a = np.sort(a,axis=0)
#
# save_file = os.path.join(d, 'num_per_class_accuracy.csv')
# a.tofile('foo.csv',sep=',',format='%10.5f')
# with open(save_file,'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(['num_per_class', 'accuracy'])
#     for a0 in a:
#         writer.writerow(['{:03d}'.format(int(a0[0])), '{:4f}'.format(a0[1])])
# l = '36dbc28567fa9275ca4fe644e2cde6a7'
# l = '297dab3ce756ca064a1aac96811903f7' # 20 label results
# d = './logs_test2/finetune'
# l = 'f686f4e38ecab012ed007e354ea346d8'
# print_score(d, l)

# with os.scandir(d) as entries:
#     for entry in entries:
#         if entry.is_dir():
#             l = entry.name
#             print_score(d, l)