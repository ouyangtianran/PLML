import pickle as pkl
import json
import os
import csv
import numpy as np

fold_name = 'ad739d780d8f2e6f0b825cefee967044'


def print_score(root, fold_name):
    file_name = '{}/{}/score_list.pkl'.format(root, fold_name)

    json_name = '{}/{}/exp_dict.json'.format(root, fold_name)

    # print(fold_name)

    with open(json_name) as f:
        js = json.load(f)
    # if 'tiered' not in js['dataset_train_root']:
    #     return 0
    print(fold_name)
    print(js)
    names = js.keys()
    print_names = ['num_per_class']
    for pn in print_names:
        if pn in names:
            print('{}: {}, '.format(pn, js[pn]))
    print(' lr {} data_root {} model {}'.format(js['lr'], js['dataset_train_root'], js['model']))

    with open(file_name, 'rb') as infile:
        data = pkl.load(infile)
    acc_max = 0
    for d in data:
        e = d['epoch']
        acc = d['test_accuracy']
        val_acc = d['val_accuracy']
        names = d.keys()
        # dataset_root = d['dataset_test_root']
        if 'test_confidence' in names:
            con = d['test_confidence']
        else:
            con = 0


        # print('epcoh: {}, test_accuracy: {}, test_confidence: {}'.format(e,acc, con))
        if val_acc > acc_max:
            ss = 'epcoh: {}, val_accuracy: {}, test_accuracy: {}, test_confidence: {}'.format(e, val_acc, acc, con)
            acc_max = val_acc
            test_acc = acc
    print('max: {}'.format(ss))

    return js['num_per_class'], test_acc




d = '/home/xd1/code/EP-semi/logs_semi_base_norm/pretraining'
d = '/home/xd1/code/EP-semi/logs_semi_base_norm/finetune'
# d = '/home/xd1/code/EP-semi/logs_semi_base/finetune'
# d = '/home/xd1/code/EP-semi/logs_emb_semi/pretraining'
# d = '/home/xd1/code/EP-semi/logs_emb_semi/finetune'
# d = '/home/xd1/code/embedding-propagation/logs/pretraining'
# d = '/home/xd1/code/EP-semi/logs/ssl'
# ll = os.listdir(d)
acc_list=[]
with os.scandir(d) as entries:
    for entry in entries:
        if entry.is_dir():
            l = entry.name
            npc, acc = print_score(d, l)
            acc_list.append([npc, acc])
            # print(entry.name)
# for l in ll:
#     npc, acc = print_score(d, l)
#     acc_list.append([npc, acc])
a = np.array(acc_list)
a = np.sort(a,axis=0)

save_file = os.path.join(d, 'num_per_class_accuracy.csv')
with open(save_file,'w') as f:
    writer = csv.writer(f)
    writer.writerow(['num_per_class', 'accuracy'])
    for a0 in a:
        writer.writerow(['{:03d}'.format(int(a0[0])), '{:4f}'.format(a0[1])])