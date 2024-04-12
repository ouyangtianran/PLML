import os
from haven import haven_utils as hu

import os
from haven import haven_utils as hu

conv4 = {
    "name": "ssl",
    'backbone':'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

wrn = {
    "name": "ssl",
    "backbone": 'wrn',
    "depth": 28,
    "width": 10,
    "transform_train": "wrn_finetune_train",
    "transform_val": "wrn_val",
    "transform_test": "wrn_val"
}

resnet12 = {
    "name": "ssl",
    "backbone": 'resnet12',
    "depth": 12,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

miniimagenet = {
    "dataset": "miniimagenet",
    "dataset_train": "episodic_miniimagenet_pkl",
    "dataset_val": "episodic_miniimagenet_pkl",
    "dataset_test": "episodic_miniimagenet_pkl",
    "data_root": "mini-imagenet",
    "n_classes": 64
}

tiered_imagenet = {
    "dataset": "tiered-imagenet",
    "n_classes": 351,
    "dataset_train": "episodic_tiered-imagenet",
    "dataset_val": "episodic_tiered-imagenet",
    "dataset_test": "episodic_tiered-imagenet",
    'data_root':'tiered-imagenet'
}

cub = {
    "dataset": "cub",
    "n_classes": 100,
    "dataset_train": "episodic_cub",
    "dataset_val": "episodic_cub",
    "dataset_test": "episodic_cub",
    'data_root':'CUB_200_2011'
}

conv4semi_test = {
    "name": "semi_ssl",
    'backbone': 'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "imagnet_nomalize",
    "transform_val": "imagnet_nomalize",
    "transform_test": "imagnet_nomalize"
}
conv4semi_test_basic = {
    "name": "semi_ssl",
    'backbone': 'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}


wrn_semi_norm = {
    "name": "semi_ssl",
    "backbone": 'wrn',
    "depth": 28,
    "width": 10,
    "transform_train": "wrn_pretrain_train_im_norm",
    "transform_val": "wrn_im_norm_val",
    "transform_test": "wrn_im_norm_val"
}

resnet12_semi = {
    "name": "semi_ssl",
    "backbone": 'resnet12',
    "depth": 12,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

resnet12_semi_norm = {
    "name": "semi_ssl",
    "backbone": 'resnet12',
    "depth": 12,
    "width": 1,
    "transform_train": "imagnet_nomalize",
    "transform_val": "imagnet_nomalize",
    "transform_test": "imagnet_nomalize"
}
conv4semi_inv = {
    "name": "semi_ssl",
    'backbone': 'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "imagnet_nomalize_inv",
    "transform_val": "imagnet_nomalize_inv",
    "transform_test": "imagnet_nomalize_inv"
}

resnet12_semi_inv = {
    "name": "semi_ssl",
    "backbone": 'resnet12',
    "depth": 12,
    "width": 1,
    "transform_train": "imagnet_nomalize_inv",
    "transform_val": "imagnet_nomalize_inv",
    "transform_test": "imagnet_nomalize_inv"
}

vit_base_semi_norm = {
    "name": "semi_ssl",
    "backbone": 'vit_base',
    "depth": 12,
    "width": 1,
    "transform_train": "imagnet_nomalize224",
    "transform_val": "imagnet_nomalize224",
    "transform_test": "imagnet_nomalize224"
}


semi_miniimagenet = {
    "dataset": "semi_miniimagenet",
    "dataset_train": "semi_episodic_miniimagenet_pkl",
    "dataset_train_test": "semi_rotated_miniimagenet_pkl",
    "dataset_val": "episodic_miniimagenet_pkl",
    "dataset_test": "episodic_miniimagenet_pkl",
    "n_classes": 64,
    "data_root": "mini-imagenet"
}


semi_tiered_imagenet = {
    "dataset": "tiered-imagenet",
    "n_classes": 351,
    "dataset_train": "semi_episodic_tiered-imagenet",
    "dataset_train_test": "semi_tiered-imagenet",
    "dataset_val": "episodic_tiered-imagenet",
    "dataset_test": "episodic_tiered-imagenet",
    "data_root": "tiered-imagenet",
}

semi_cub = {
    "dataset": "cub",
    "n_classes": 100,
    "dataset_train": "semi_episodic_cub",
    "dataset_train_test": "semi_cub",
    "dataset_val": "episodic_cub",
    "dataset_test": "episodic_cub",
    "data_root": "CUB_200_2011"
}

EXP_GROUPS = {}
EXP_GROUPS['ssl_large'] = []
embedding_prop = True
# 12 exps
for dataset in [semi_miniimagenet]:#[miniimagenet, tiered_imagenet]:
    for backbone in [conv4semi_test, resnet12_semi_norm]:#[resnet12, conv4, wrn]:
        # for embedding_prop in [True]:
        for npc in [20, 50, 100]:
            for shot in [1,5]:#[1, 5]:
                EXP_GROUPS['ssl_large'] += [{
                    'dataset_train_root': dataset["data_root"],
                    'dataset_val_root': dataset["data_root"],
                    'dataset_test_root': dataset["data_root"],
                    "model": backbone,
                        
                    # Hardware
                    "ngpu": 1,
                    "random_seed": 42,

                    # Optimization
                    "batch_size": 1,
                    "train_iters": 10,
                    "test_iters": 600,
                    "tasks_per_batch": 1,

                    # Model
                    "dropout": 0.1,
                    "avgpool": True,

                    # Data
                    'n_classes': dataset["n_classes"],
                    "collate_fn": "identity",
                    "transform_train": backbone["transform_train"],

                    "transform_val": backbone["transform_val"],
                    "transform_test": backbone["transform_test"],
                    "dataset_train": dataset["dataset_train"],
                    "classes_train": 5,
                    "support_size_train": shot,
                    "query_size_train": 15,
                    "unlabeled_size_train": 0,

                    "dataset_val": dataset["dataset_val"],
                    "classes_val": 5,
                    "support_size_val": shot,
                    "query_size_val": 15,
                    "unlabeled_size_val": 0,

                    "dataset_test": dataset["dataset_test"],
                    "classes_test": 5,
                    "support_size_test": shot,
                    "query_size_test": 15,
                    "unlabeled_size_test": 100,
                    "predict_method": "labelprop",
                    "finetuned_weights_root": "logs_noise_v12/finetune", #"./logs_semi_base_semco_v3/finetune_cw01",

                    # Hparams
                    "embedding_prop" : embedding_prop,
                    # semi dataset
                    "num_per_class": npc,
                    "rotation_labels": [0],
                    "unsupervised_loss": "semco",
                    "semi_batch_size": 128,
                    "unlabeled_multiple": 0,
                    "dataset_train_test": dataset["dataset_train_test"],
                    "max_epoch": 1,

                    # word embedding
                    "word_emb_weight": 0,
                    "adjust_type": "nolinear",
                    "adjust_dropout": 0.7,
                    # noise labels
                    "trans_encoder": True,  # True, #te,#True

                    }]
EXP_GROUPS["ssl_reb"]=[]
base_setting = EXP_GROUPS['ssl_large'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [50, 100]:#[-1, 20]:#[20, 100, 200]:#[20, 60, 120, 300, 540]:
                for shot in [1, 5]:  # [1, 5]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        # "lr": 0.001,
                        # "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",

                        # Data
                        'n_classes': dataset["n_classes"],
                        "transform_train": backbone["transform_train"],
                        "transform_val": backbone["transform_val"],
                        "transform_test": backbone["transform_test"],
                        "dataset_train": dataset["dataset_train"],
                        "dataset_train_test": dataset["dataset_train_test"],
                        'dataset_train_root': dataset["data_root"],

                        "dataset_val": dataset["dataset_val"],
                        'dataset_val_root': dataset["data_root"],

                        "dataset_test": dataset["dataset_test"],
                        'dataset_test_root': dataset["data_root"],
                        # semi dataset
                        "num_per_class": npc,
                        "finetuned_weights_root": "logs_iccv_re/finetune_trans_encoder_resort3",#"./logs_semi_base_semco_v3/finetune_cw01_tiered",
                        "support_size_train": shot,
                        "support_size_val": shot,
                        "support_size_test": shot,
                        "trans_encoder": True,
                        "embedding_prop": True,
                        "n_distract":20
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['ssl_reb'] += [new_setting]

EXP_GROUPS["ssl_all"]=[]
base_setting = EXP_GROUPS['ssl_large'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [resnet12_semi_norm]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [-1]:#[-1, 20]:#[20, 100, 200]:#[20, 60, 120, 300, 540]:
                for shot in [1, 5]:  # [1, 5]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        # "lr": 0.001,
                        # "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",

                        # Data
                        'n_classes': dataset["n_classes"],
                        "transform_train": backbone["transform_train"],
                        "transform_val": backbone["transform_val"],
                        "transform_test": backbone["transform_test"],
                        "dataset_train": dataset["dataset_train"],
                        "dataset_train_test": dataset["dataset_train_test"],
                        'dataset_train_root': dataset["data_root"],

                        "dataset_val": dataset["dataset_val"],
                        'dataset_val_root': dataset["data_root"],

                        "dataset_test": dataset["dataset_test"],
                        'dataset_test_root': dataset["data_root"],
                        # semi dataset
                        "num_per_class": npc,
                        "finetuned_weights_root": "logs_iccv_re/finetune_all",#"./logs_semi_base_semco_v3/finetune_cw01_tiered",
                        "support_size_train": shot,
                        "support_size_val": shot,
                        "support_size_test": shot,
                        "trans_encoder": True,
                        "embedding_prop": True,
                        "n_distract":0
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['ssl_all'] += [new_setting]

EXP_GROUPS["ssl_reb_npc20"]=[]
base_setting = EXP_GROUPS['ssl_large'][0]
npc=20
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            # for npc in [50, 100]:#[-1, 20]:#[20, 100, 200]:#[20, 60, 120, 300, 540]:
            for nd in [30, 64]:
                for shot in [1, 5]:  # [1, 5]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        # "lr": 0.001,
                        # "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",

                        # Data
                        'n_classes': dataset["n_classes"],
                        "transform_train": backbone["transform_train"],
                        "transform_val": backbone["transform_val"],
                        "transform_test": backbone["transform_test"],
                        "dataset_train": dataset["dataset_train"],
                        "dataset_train_test": dataset["dataset_train_test"],
                        'dataset_train_root': dataset["data_root"],

                        "dataset_val": dataset["dataset_val"],
                        'dataset_val_root': dataset["data_root"],

                        "dataset_test": dataset["dataset_test"],
                        'dataset_test_root': dataset["data_root"],
                        # semi dataset
                        "num_per_class": npc,
                        "finetuned_weights_root": "logs_iccv_re/finetune_npc20",#"./logs_semi_base_semco_v3/finetune_cw01_tiered",
                        "support_size_train": shot,
                        "support_size_val": shot,
                        "support_size_test": shot,
                        "trans_encoder": True,
                        "embedding_prop": True,
                        "n_distract":nd
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['ssl_reb_npc20'] += [new_setting]

EXP_GROUPS["ssl_tiered"]=[]
base_setting = EXP_GROUPS['ssl_large'][0]
for dataset in [semi_tiered_imagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test, resnet12_semi_norm]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 100, 200]:#[20, 60, 120, 300, 540]:
                for shot in [1, 5]:  # [1, 5]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        # "ngpu": 1,
                        # "lr": 0.001,
                        # "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",

                        # Data
                        'n_classes': dataset["n_classes"],
                        "transform_train": backbone["transform_train"],
                        "transform_val": backbone["transform_val"],
                        "transform_test": backbone["transform_test"],
                        "dataset_train": dataset["dataset_train"],
                        "dataset_train_test": dataset["dataset_train_test"],
                        'dataset_train_root': dataset["data_root"],

                        "dataset_val": dataset["dataset_val"],
                        'dataset_val_root': dataset["data_root"],

                        "dataset_test": dataset["dataset_test"],
                        'dataset_test_root': dataset["data_root"],
                        # semi dataset
                        "num_per_class": npc,
                        "finetuned_weights_root": "logs_noise_v12/finetune_tiered",#"./logs_semi_base_semco_v3/finetune_cw01_tiered",
                        "support_size_train": shot,
                        "support_size_val": shot,
                        "support_size_test": shot,
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['ssl_tiered'] += [new_setting]

EXP_GROUPS["flexmatch_ssl"]=[]
base_setting = EXP_GROUPS['ssl_large'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_inv, resnet12_semi_inv]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 50, 100]:#[20, 60, 120, 300, 540]:
                for shot in [1, 5]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        # "ngpu": 1,
                        # "lr": 0.001,
                        # "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",

                        # Data
                        'n_classes': dataset["n_classes"],
                        "transform_train": backbone["transform_train"],
                        "transform_val": backbone["transform_val"],
                        "transform_test": backbone["transform_test"],
                        "dataset_train": dataset["dataset_train"],
                        "dataset_train_test": dataset["dataset_train_test"],
                        'dataset_train_root': dataset["data_root"],

                        "dataset_val": dataset["dataset_val"],
                        'dataset_val_root': dataset["data_root"],

                        "dataset_test": dataset["dataset_test"],
                        'dataset_test_root': dataset["data_root"],
                        # semi dataset
                        "num_per_class": npc,
                        "finetuned_weights_root": "logs_noise_v12/flexmatch_finetune",#"./logs_flexmatch/finetune_cw01",
                        "support_size_train": shot,
                        "support_size_val": shot,
                        "support_size_test": shot,
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['flexmatch_ssl'] += [new_setting]

EXP_GROUPS["flexmatch_ssl_tiered"]=[]
base_setting = EXP_GROUPS['ssl_large'][0]
for dataset in [semi_tiered_imagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_inv, resnet12_semi_inv]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 100, 200]:#[20, 60, 120, 300, 540]:
                for shot in [1, 5]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        # "ngpu": 1,
                        # "lr": 0.001,
                        # "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",

                        # Data
                        'n_classes': dataset["n_classes"],
                        "transform_train": backbone["transform_train"],
                        "transform_val": backbone["transform_val"],
                        "transform_test": backbone["transform_test"],
                        "dataset_train": dataset["dataset_train"],
                        "dataset_train_test": dataset["dataset_train_test"],
                        'dataset_train_root': dataset["data_root"],

                        "dataset_val": dataset["dataset_val"],
                        'dataset_val_root': dataset["data_root"],

                        "dataset_test": dataset["dataset_test"],
                        'dataset_test_root': dataset["data_root"],
                        # semi dataset
                        "num_per_class": npc,
                        "finetuned_weights_root": "logs_noise_v12/flexmatch_finetune_tiered",#"./logs_flexmatch/finetune_cw01_tiered",
                        "support_size_train": shot,
                        "support_size_val": shot,
                        "support_size_test": shot,
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['flexmatch_ssl_tiered'] += [new_setting]

shot=1
query_shot = 1
EXP_GROUPS["ssl_cub"]=[]
base_setting = EXP_GROUPS['ssl_large'][0]
for dataset in [semi_cub]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test, resnet12_semi_norm]:#[conv4semi_test, resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [2, 6, 10]:#[2, 6, 10]:#[20, 60, 120, 300, 540]:
                for shot in [1, 5]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_cub",
                        # "pretrained_weights_root": "./logs_test/pretraining",

                        # Data
                        'n_classes': dataset["n_classes"],
                        "transform_train": backbone["transform_train"],
                        "transform_val": backbone["transform_val"],
                        "transform_test": backbone["transform_test"],
                        "dataset_train": dataset["dataset_train"],
                        "dataset_train_test": dataset["dataset_train_test"],
                        'dataset_train_root': dataset["data_root"],

                        "dataset_val": dataset["dataset_val"],
                        'dataset_val_root': dataset["data_root"],

                        "dataset_test": dataset["dataset_test"],
                        'dataset_test_root': dataset["data_root"],
                        # semi dataset
                        "num_per_class": npc,
                        "support_size_train": shot,
                        "query_size_train": query_shot,
                        "support_size_val": shot,
                        "query_size_val": 15,
                        "support_size_test": shot,
                        "query_size_test": 15,
                        "finetuned_weights_root": "./logs_semi_base_semco_v3/finetune_cw01_cub",
                        "unlabeled_size_test": 20,
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['ssl_cub'] += [new_setting]


EXP_GROUPS["flexmatch_ssl_cub"]=[]
base_setting = EXP_GROUPS['ssl_large'][0]
for dataset in [semi_cub]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_inv, resnet12_semi_inv]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [2, 6, 10]:#[20, 60, 120, 300, 540]:
                for shot in [1, 5]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        # "ngpu": 1,
                        # "lr": 0.001,
                        # "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",

                        # Data
                        'n_classes': dataset["n_classes"],
                        "transform_train": backbone["transform_train"],
                        "transform_val": backbone["transform_val"],
                        "transform_test": backbone["transform_test"],
                        "dataset_train": dataset["dataset_train"],
                        "dataset_train_test": dataset["dataset_train_test"],
                        'dataset_train_root': dataset["data_root"],

                        "dataset_val": dataset["dataset_val"],
                        'dataset_val_root': dataset["data_root"],

                        "dataset_test": dataset["dataset_test"],
                        'dataset_test_root': dataset["data_root"],
                        # semi dataset
                        "num_per_class": npc,
                        "finetuned_weights_root": "./logs_flexmatch/finetune_cw01_cub",
                        "support_size_train": shot,
                        "query_size_train": query_shot,
                        "support_size_val": shot,
                        "query_size_val": 15,
                        "support_size_test": shot,
                        "query_size_test": 15,
                        "unlabeled_size_test": 20,
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['flexmatch_ssl_cub'] += [new_setting]

shot=5
query_shot = 15
EXP_GROUPS["ssl_cub_5s15q"]=[]
base_setting = EXP_GROUPS['ssl_large'][0]
for dataset in [semi_cub]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test, resnet12_semi_norm]:#[conv4semi_test, resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [2, 6, 10]:#[2, 6, 10]:#[20, 60, 120, 300, 540]:
                for shot in [1, 5]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_cub",
                        # "pretrained_weights_root": "./logs_test/pretraining",

                        # Data
                        'n_classes': dataset["n_classes"],
                        "transform_train": backbone["transform_train"],
                        "transform_val": backbone["transform_val"],
                        "transform_test": backbone["transform_test"],
                        "dataset_train": dataset["dataset_train"],
                        "dataset_train_test": dataset["dataset_train_test"],
                        'dataset_train_root': dataset["data_root"],

                        "dataset_val": dataset["dataset_val"],
                        'dataset_val_root': dataset["data_root"],

                        "dataset_test": dataset["dataset_test"],
                        'dataset_test_root': dataset["data_root"],
                        # semi dataset
                        "num_per_class": npc,
                        "support_size_train": shot,
                        "query_size_train": query_shot,
                        "support_size_val": shot,
                        "query_size_val": 15,
                        "support_size_test": shot,
                        "query_size_test": 15,
                        "finetuned_weights_root": "./logs_cub_5w5s/finetune_cw01_cub_lr0001_1gpu",
                        "unlabeled_size_test": 20,
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['ssl_cub_5s15q'] += [new_setting]


EXP_GROUPS["flexmatch_ssl_cub_5s15q"]=[]
base_setting = EXP_GROUPS['ssl_large'][0]
for dataset in [semi_cub]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_inv, resnet12_semi_inv]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [2, 6, 10]:#[20, 60, 120, 300, 540]:
                for shot in [1, 5]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        # "ngpu": 1,
                        # "lr": 0.001,
                        # "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",

                        # Data
                        'n_classes': dataset["n_classes"],
                        "transform_train": backbone["transform_train"],
                        "transform_val": backbone["transform_val"],
                        "transform_test": backbone["transform_test"],
                        "dataset_train": dataset["dataset_train"],
                        "dataset_train_test": dataset["dataset_train_test"],
                        'dataset_train_root': dataset["data_root"],

                        "dataset_val": dataset["dataset_val"],
                        'dataset_val_root': dataset["data_root"],

                        "dataset_test": dataset["dataset_test"],
                        'dataset_test_root': dataset["data_root"],
                        # semi dataset
                        "num_per_class": npc,
                        "finetuned_weights_root": "./logs_cub_5w5s/flexmatch_finetune_cub_lr0001",
                        "support_size_train": shot,
                        "query_size_train": query_shot,
                        "support_size_val": shot,
                        "query_size_val": 15,
                        "support_size_test": shot,
                        "query_size_test": 15,
                        "unlabeled_size_test": 20,
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['flexmatch_ssl_cub_5s15q'] += [new_setting]

# 24 exps
EXP_GROUPS['ssl_small'] = []
for dataset in [miniimagenet]:#[tiered_imagenet, miniimagenet]:
    for backbone in [conv4]:#[[conv4, resnet12, wrn]:
        for embedding_prop in [True]:
                for shot, ust in zip([1,2,3], 
                                     [4,3,2]):
                    EXP_GROUPS['ssl_small'] += [{
                        'dataset_train_root': dataset["data_root"],
                        'dataset_val_root': dataset["data_root"],
                        'dataset_test_root': dataset["data_root"],

                        "model": backbone,
                            
                            # Hardware
                            "ngpu": 1,
                            "random_seed": 42,

                            # Optimization
                            "batch_size": 1,
                            "train_iters": 10,
                            "val_iters": 600,
                            "test_iters": 600,
                            "tasks_per_batch": 1,

                            # Model
                            "dropout": 0.1,
                            "avgpool": True,

                            # Data
                            'n_classes': dataset["n_classes"],
                            "collate_fn": "identity",
                            "transform_train": backbone["transform_train"],
                            "transform_val": backbone["transform_val"],
                            "transform_test": backbone["transform_test"],

                            "dataset_train": dataset["dataset_train"],
                            "classes_train": 5,
                            "support_size_train": shot,
                            "query_size_train": 15,
                            "unlabeled_size_train": 0,

                            "dataset_val": dataset["dataset_val"],
                            "classes_val": 5,
                            "support_size_val": shot,
                            "query_size_val": 15,
                            "unlabeled_size_val": 0,

                            "dataset_test": dataset["dataset_test"],
                            "classes_test": 5,
                            "support_size_test": shot,
                            "query_size_test": 15,
                            "unlabeled_size_test": ust,
                            "predict_method":"labelprop",
                            "finetuned_weights_root": "./logs_emb/finetune",

                            # Hparams
                            "embedding_prop" : embedding_prop,

                            # word embedding
                            "word_emb_weight": 1,
                            "adjust_type": "nolinear",
                            "adjust_dropout": 0.7
                        }]

EXP_GROUPS["marginmatch_ssl"]=[]
base_setting = EXP_GROUPS['ssl_large'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_inv, resnet12_semi_inv]:#[conv4semi_inv, resnet12_semi_inv][ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 50, 100]:#[20, 50, 100][20, 60, 120, 300, 540]:
                for shot in [1, 5]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        # "ngpu": 1,
                        # "lr": 0.001,
                        # "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",

                        # Data
                        'n_classes': dataset["n_classes"],
                        "transform_train": backbone["transform_train"],
                        "transform_val": backbone["transform_val"],
                        "transform_test": backbone["transform_test"],
                        "dataset_train": dataset["dataset_train"],
                        "dataset_train_test": dataset["dataset_train_test"],
                        'dataset_train_root': dataset["data_root"],

                        "dataset_val": dataset["dataset_val"],
                        'dataset_val_root': dataset["data_root"],

                        "dataset_test": dataset["dataset_test"],
                        'dataset_test_root': dataset["data_root"],
                        # semi dataset
                        "num_per_class": npc,
                        "finetuned_weights_root": "logs_noise_v12/marginmatch_finetune",#"./logs_flexmatch/finetune_cw01",
                        "support_size_train": shot,
                        "support_size_val": shot,
                        "support_size_test": shot,
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['marginmatch_ssl'] += [new_setting]
