import pickle as pkl
import json
import os
import pandas as pd

fold_name = 'ad739d780d8f2e6f0b825cefee967044'


def print_score(root, fold_name, logger):
    file_name = '{}/{}/score_list.pkl'.format(root, fold_name)

    json_name = '{}/{}/exp_dict.json'.format(root, fold_name)

    # print(fold_name)

    with open(json_name) as f:
        js = json.load(f)
    # if 'tiered' not in js['dataset_train_root']:
    #     return 0
    # print(fold_name)
    # print(js)
    names = js.keys()
    print_names = ['num_per_class',"unsupervised_loss","unlabeled_multiple","rotation_weight","pseudo_thr",
                   "pseudo_thr_g", "lr", "semi_batch_size","use_unlabeled_data", "delay_iter", "freeze"]
    print_names = ["num_per_class","lr", "classification_weight","embedding_prop_clc","unlabeled_multiple","pseudo_thr_g","unlabeled_size_train","model"]#
    print_str = ""#,
    # if js["num_per_class"]== 20:
    for pn in print_names:
        if pn in names:
            print_str += '{}: {}, '.format(pn, js[pn])
            # print('{}: {}, '.format(pn, js[pn]))
    # print(print_str)
    # print(' lr {} data_root {} model {}'.format(js['lr'], js['dataset_train_root'], js['model']))

    with open(file_name, 'rb') as infile:
        data = pkl.load(infile)

        # Report
        score_list = data
        score_df = pd.DataFrame(score_list)
        # print(score_df.tail())

    acc_max = 0
    print_names_acc = ['epoch', 'test_accuracy', 'test_confidence', 'val_accuracy', 'train_test_accuracy']
    for d in data:
        val_acc = d['val_accuracy']
        if val_acc > acc_max:
            # ss = 'epcoh: {}, val_accuracy: {},test_accuracy: {}, test_confidence: {}'.format(e, val_acc, acc, con)
            names = d.keys()
            print_str_acc = ""
            for pn in print_names_acc:
                if pn in names:
                    print_str_acc += '{}: {}, '.format(pn, d[pn])
            ss = print_str_acc
            acc_max = val_acc

    if 1:#js["model"]["backbone"] == 'conv4':#js["num_per_class"]== 20:#1:#js["unlabeled_multiple"] == 1:#js["num_per_class"]== 20:#js["lr"] == 0.2:
        # resnet12, wrn
        logger.info(fold_name)
        logger.info(print_str)
        logger.info('max: {}'.format(ss))
    # print accuracy_all_data for pseudo-labels
    logger.info('accuracy_all_data for pseudo-labels: {}'.format(score_df['accuracy_all_data'][0]))




d = './logs_semi_base_norm_low_shot/pretraining'
d = './logs_semi_base_semco_low_shot/pretraining'
# d = './logs_semi_base_semco_ab/pretraining_test'
# d = './logs_semi_base_semco_ab/finetune_best_pre_cw'
d = './logs_semi_base_semco/finetune_dy_pseudo_thr_g'
d = './logs_semi_base_semco/finetune_semco_iter600'
# d = './logs_semi_base_semco/pretraining'
d = './logs_semi_base_norm/finetune3'
d = './logs_semi_base_semco_v2/finetune_thr_g'
d = './logs_semi_base_semco_v3/finetune_lr'
d = './logs_semi_base_norm/pretrain_cub'
d = './logs_semi_base_semco_v3/finetune_tiered'
d = './logs_semi_base_norm/pretrain_tiered'
d='/home/xd1/code/EP-semi/logs_flexmatch/finetune_cw01_cub_lr0001'
d='/home/xd1/code/EP-semi/logs_flexmatch/proto_cw01_cub_lr0001_ep'
d='/home/xd1/code/EP-semi/logs_flexmatch/proto_cw01_cub_lr0001_1gpu'
d='/home/xd1/code/EP-semi/logs_flexmatch/proto_cw01_cub_lr0001_ep_1gpu'
d='/home/xd1/code/EP-semi/logs_semi_base_semco_proto/proto_cw01_cub_lr0001_1gpu'
d='/home/xd1/code/EP-semi/logs_semi_base_semco_proto/proto_cw01_cub_lr0001_ep_1gpu'
d='./logs_flexmatch/proto_cw01_cub_lr0001_1gpu'
d='/home/xd1/code/EP-semi/logs_flexmatch/proto_cw01_tiered'
d='/home/xd1/code/EP-semi/logs_semi_base_semco_proto/finetune_tiered_cw01'
d='/home/xd1/code/EP-semi/logs_semi_base_semco_v3/finetune_cw01_tiered_lr0001_true'
d='/home/xd1/code/EP-semi/logs_flexmatch/finetune_cw01'
d='/home/xd1/code/EP-semi/logs_noise_semco_v01/finetune_cw01'
# d='/home/xd1/code/EP-semi/logs_flexmatch/proto_cw01_cub_lr0001'
# d = './logs_semi_base_norm/finetune_cub'
# d = './logs_semi_base_semco_v3/finetune_hp_thr_g'
# d='/home/xd1/code/EP-semi/logs_semi_base_semco_proto/finetune_tiered_cw01'
# d = './logs/pretraining'
# d = '/home/xd1/code/EP-semi/logs_semi_base/finetune'
# d = './logs_semi_base_norm/finetune3_iter600'
# d = './logs_test/pretraining'
# d = './logs_emb_semi/finetune'
# d = '/home/xd1/code/EP-semi/logs_emb_semi/finetune'
# d = '/home/xd1/code/embedding-propagation/logs/pretraining'
# d = '/home/xd1/code/EP-semi/logs/ssl'

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
# set log file path
handler = logging.FileHandler(os.path.join(d, "log_scores.txt"))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)

# logger.info("Start print log")
# logger.debug("Do something")
# logger.warning("Something maybe fail.")
# logger.info("Finish")

ll = os.listdir(d)
logger.info(d)
for l in ll:
    if '.' in l:
        continue
    print_score(d, l, logger)

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