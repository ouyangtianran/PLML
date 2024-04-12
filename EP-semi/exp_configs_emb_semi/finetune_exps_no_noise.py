 

conv4 = {
    "name": "finetuning",
    'backbone':'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

conv4semi_test = {
    "name": "semi_finetuning",
    'backbone': 'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "imagnet_nomalize",
    "transform_val": "imagnet_nomalize",
    "transform_test": "imagnet_nomalize"
}

conv4noise = {
    "name": "noise_finetuning",
    'backbone': 'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "imagnet_nomalize",
    "transform_val": "imagnet_nomalize",
    "transform_test": "imagnet_nomalize"
}

resnet12_noise = {
    "name": "noise_finetuning",
    "backbone": 'resnet12',
    "depth": 12,
    "width": 1,
    "transform_train": "imagnet_nomalize",
    "transform_val": "imagnet_nomalize",
    "transform_test": "imagnet_nomalize"
}

conv4semi_test_basic = {
    "name": "semi_finetuning",
    'backbone': 'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

wrn = {
    "name": "finetuning",
    "backbone": 'wrn',
    "depth": 28,
    "width": 10,
    "transform_train": "wrn_finetune_train",
    "transform_val": "wrn_val",
    "transform_test": "wrn_val"
}

wrn_semi_norm = {
    "name": "semi_finetuning",
    "backbone": 'wrn',
    "depth": 28,
    "width": 10,
    "transform_train": "wrn_pretrain_train_im_norm",
    "transform_val": "wrn_im_norm_val",
    "transform_test": "wrn_im_norm_val"
}

resnet12_semi = {
    "name": "semi_finetuning",
    "backbone": 'resnet12',
    "depth": 12,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

resnet12_semi_norm = {
    "name": "semi_finetuning",
    "backbone": 'resnet12',
    "depth": 12,
    "width": 1,
    "transform_train": "imagnet_nomalize",
    "transform_val": "imagnet_nomalize",
    "transform_test": "imagnet_nomalize"
}

vit_base_semi_norm = {
    "name": "semi_finetuning",
    "backbone": 'vit_base',
    "depth": 12,
    "width": 1,
    "transform_train": "imagnet_nomalize224",
    "transform_val": "imagnet_nomalize224",
    "transform_test": "imagnet_nomalize224"
}

resnet12 = {
    "name": "finetuning",
    "backbone": 'resnet12',
    "depth": 12,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

conv4semi_inv = {
    "name": "semi_finetuning",
    'backbone': 'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "imagnet_nomalize_inv",
    "transform_val": "imagnet_nomalize_inv",
    "transform_test": "imagnet_nomalize_inv"
}

resnet12_semi_inv = {
    "name": "semi_finetuning",
    "backbone": 'resnet12',
    "depth": 12,
    "width": 1,
    "transform_train": "imagnet_nomalize_inv",
    "transform_val": "imagnet_nomalize_inv",
    "transform_test": "imagnet_nomalize_inv"
}

miniimagenet = {
    "dataset": "miniimagenet",
    "dataset_train": "episodic_miniimagenet_pkl",
    "dataset_val": "episodic_miniimagenet_pkl",
    "dataset_test": "episodic_miniimagenet_pkl",
    "data_root": "mini-imagenet",
    "n_classes": 64
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

tiered_imagenet = {
    "dataset": "tiered-imagenet",
    "n_classes": 351,
    "dataset_train": "episodic_tiered-imagenet",
    "dataset_val": "episodic_tiered-imagenet",
    "dataset_test": "episodic_tiered-imagenet",
    "data_root": "tiered-imagenet",
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

cub = {
    "dataset": "cub",
    "n_classes": 100,
    "dataset_train": "episodic_cub",
    "dataset_val": "episodic_cub",
    "dataset_test": "episodic_cub",
    "data_root": "CUB_200_2011"
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

EXP_GROUPS = {"finetune": []}
lr=0.01
shot = 5
dropout = 0.7
way=5
query_shot = 15
un_mu = 0
p_thr = 0
p_thr_g = 0#0.95
npc=50
delay_iter = -1
freeze = 0
train_iters = 100 # 100 600
unlabeled_size_train = 0 #0,16, 12
pretrain_method = 'semco'
embedding_prop_clc = False#True#False
# keep_labeled_data = True
un_mu_bs = [[0,128]]#[[1,64],[3,32],[7,16],[0,128]]#
use_unlabeled_data = "only_pseudo"#"only_pseudo"  # "no_iteration",#"iterating_optimization", #""
weight_loss = False
for dataset in [semi_miniimagenet]:#[semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test, resnet12_semi_norm]:#[conv4semi_test,wrn_semi_norm,resnet12_semi_norm,conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
        # for lr in [0.01, 0.001]:#              [0.1, 0.02, 0.01, 0.008, 0.005, 0.002]:
        # for dropout in [0.3, 0.5]:
        # for way in [5]:
        # for delay_iter in [5, 10,20,30]:#[-1, 0]
        # for freeze in [0]:#[4,3,2,1]:#[4,3,2,1,0]
        # for p_thr_g in [-1, -2]:
        # for use_unlabeled_data in [""]:
           for un_mu, bs in un_mu_bs:
            # for p_thr in [ 0.1, 0.5, 0.7, 0.95]:
            # for p_thr_g in [0]:#[0.8, 0.7, 0.5, 0.1, 0]:#[0.95, 0.7, 0.5, 0.1,0]:#-1, -2, [0.1, 0.5, 0.7, 0.95, 0]:
        #     for shot in [1]:  # [1]:#
        #     for embedding_prop_clc in [True, False]:
              for npc in [20, 50, 100]:#[20]:#[20, 60, 120, 300, 540]:
                # for train_iters in [100]:  # [100, 600]:
                # for unlabeled_size_train in [4,8,12]:
                    for classification_weight in [0.1]:#[0, 0.05, 0.1, 0.2, 0.5, 1, 2]:#[0, 0.1, 0.5, 1, 1.5, 2, 3]:  # [0, 0.1, 0.5,1,1.5, 2, 3]:
                            EXP_GROUPS['finetune'] += [{"model": backbone,

                                                        # Hardware
                                                        "ngpu": 4,
                                                        "random_seed": 42,

                                                        # Optimization
                                                        "batch_size": 1,
                                                        "target_loss": "val_accuracy",
                                                        "lr": lr,
                                                        "min_lr_decay": 0.0001,
                                                        "weight_decay": 0.0005,
                                                        "patience": 10,
                                                        "max_epoch": 200,
                                                        "train_iters": train_iters,
                                                        "val_iters": 600,
                                                        "test_iters": 1000,
                                                        "tasks_per_batch": 1,
                                                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_test/pretraining",#
                                                                                   # "Conv4_mini-imagenet_2021-09-15_12:29:34_checkpoint_dict.pth",
                                                        "pretrain_method": pretrain_method,

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
                                                        "dataset_train_test": dataset["dataset_train_test"],
                                                        'dataset_train_root': dataset["data_root"],
                                                        "classes_train": 5,
                                                        "support_size_train": shot,
                                                        "query_size_train": query_shot,
                                                        "unlabeled_size_train": unlabeled_size_train,

                                                        "dataset_val": dataset["dataset_val"],
                                                        'dataset_val_root': dataset["data_root"],
                                                        "classes_val": 5,
                                                        "support_size_val": shot,
                                                        "query_size_val": 15,
                                                        "unlabeled_size_val": 0,

                                                        "dataset_test": dataset["dataset_test"],
                                                        'dataset_test_root': dataset["data_root"],
                                                        "classes_test": 5,
                                                        "support_size_test": shot,
                                                        "query_size_test": 15,
                                                        "unlabeled_size_test": 0,

                                                        # Hparams
                                                        "embedding_prop" : True,
                                                        "few_shot_weight": 1,
                                                        "classification_weight": classification_weight,
                                                        "rotation_weight": 0,
                                                        "active_size": 0,
                                                        "distance_type": "labelprop",
                                                        "rotation_labels": [0],

                                                        # word embedding
                                                        "word_emb_weight": 0,
                                                        "adjust_type": "linear",
                                                        "adjust_dropout": dropout,
                                                        "word_emb_trans": "",#"MDS",  # "Isomap", "Spectral", "MDS",
                                                        "word_emb_loss": "l2",

                                                        # semi dataset
                                                        "num_per_class": npc,
                                                        # "use_unlabeled_data": False,
                                                        # "remove_noise": "keep_visual",  # ["no","half",]
                                                        "unlabeled_multiple0": un_mu, # for pretraining model
                                                        "unlabeled_multiple": un_mu,  # for pretraining model
                                                        "unsupervised_loss": "semco",#"semco",
                                                        # ["simsiam","word_proto","rotation","classifier_loss"]
                                                        "lambda_emb": 0,
                                                        "pseudo_thr": p_thr,
                                                        "pseudo_thr_g": p_thr_g,
                                                        "use_unlabeled_data": use_unlabeled_data,#"iterating_optimization",#"no_iteration",#"iterating_optimization", #""
                                                        "delay_iter": delay_iter,
                                                        "semi_batch_size": bs,
                                                        "freeze":freeze,
                                                        "cross_entropy_weight": 1,
                                                        # "keep_labeled_data": keep_labeled_data
                                                        "embedding_prop_clc": embedding_prop_clc,
                                                        "weight_loss": weight_loss
                                                        }]

EXP_GROUPS["finetune_tiered"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_tiered_imagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test, resnet12_semi_norm]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 100, 200]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        # "ngpu": 1,
                        # "lr": 0.001,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
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
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['finetune_tiered'] += [new_setting]

EXP_GROUPS["finetune_tiered_lr0001"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_tiered_imagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test, resnet12_semi_norm]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 100, 200]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        # "ngpu": 1,
                        "lr": 0.001,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
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
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['finetune_tiered_lr0001'] += [new_setting]

shot=1
query_shot = 1
EXP_GROUPS["finetune_cub"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_cub]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test, resnet12_semi_norm]:#[conv4semi_test, resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [2, 6, 10]:#[2, 6, 10]:#[20, 60, 120, 300, 540]:
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
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['finetune_cub'] += [new_setting]

EXP_GROUPS["finetune_cub_lr0001"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_cub]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test]:#[conv4semi_test, resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [2, 6]:#[2, 6, 10]:#[20, 60, 120, 300, 540]:
                for lr in [0.01, 0.005, 0.0005]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        "lr":lr,
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
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['finetune_cub_lr0001'] += [new_setting]

EXP_GROUPS["proto_finetune"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test,resnet12_semi_norm]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 50, 100]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        # "ngpu": 1,
                        # "lr": 0.001,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        "embedding_prop": False,
                        "distance_type": "prototypical",

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
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['proto_finetune'] += [new_setting]

EXP_GROUPS["proto_finetune_lr0001"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 50, 100]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        # "ngpu": 1,
                        "lr": 0.001,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        "embedding_prop": False,
                        "distance_type": "prototypical",

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
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['proto_finetune_lr0001'] += [new_setting]

EXP_GROUPS["proto_finetune_tiered"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_tiered_imagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [ resnet12_semi_norm]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 100, 200]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        # "ngpu": 1,
                        # "lr": 0.001,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        "embedding_prop": False,
                        "distance_type": "prototypical",

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
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['proto_finetune_tiered'] += [new_setting]

EXP_GROUPS["proto_finetune_tiered_lr0001"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_tiered_imagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [resnet12_semi_norm, conv4semi_test]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 100, 200]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 4,
                        "lr": 0.001,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        "embedding_prop": False,
                        "distance_type": "prototypical",

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
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['proto_finetune_tiered_lr0001'] += [new_setting]

shot=1
query_shot = 1
EXP_GROUPS["proto_finetune_cub"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_cub]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test, resnet12_semi_norm]:#[conv4semi_test, resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [2, 6, 10]:#[2, 6, 10]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_cub",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        "embedding_prop": False,
                        "distance_type": "prototypical",

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
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['proto_finetune_cub'] += [new_setting]

EXP_GROUPS["flexmatch_finetune"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_inv, resnet12_semi_inv]:#[, resnet12_semi_inv conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 50, 100]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 4,
                        # "lr": 0.001,
                        "pretrained_weights_root": "../TorchSSL/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        # "embedding_prop": False,
                        # "distance_type": "prototypical",

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
                        "pretrain_method": 'flexmatch',
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['flexmatch_finetune'] += [new_setting]



EXP_GROUPS["flexmatch_finetune_tiered"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_tiered_imagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_inv, resnet12_semi_inv]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 100, 200]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 4,
                        # "lr": 0.001,
                        "pretrained_weights_root": "../TorchSSL/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        # "embedding_prop": False,
                        # "distance_type": "prototypical",

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
                        "pretrain_method": 'flexmatch',
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['flexmatch_finetune_tiered'] += [new_setting]

shot=1
query_shot = 1
EXP_GROUPS["flexmatch_finetune_cub"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_cub]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_inv, resnet12_semi_inv]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [2, 6, 10]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        # "lr": 0.001,
                        "pretrained_weights_root": "../TorchSSL/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        # "embedding_prop": False,
                        # "distance_type": "prototypical",

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
                        "pretrain_method": 'flexmatch',
                        "support_size_train": shot,
                        "query_size_train": query_shot,
                        "support_size_val": shot,
                        "query_size_val": 15,
                        "support_size_test": shot,
                        "query_size_test": 15,
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['flexmatch_finetune_cub'] += [new_setting]


EXP_GROUPS["flexmatch_proto_lr0001"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_inv]:#[, resnet12_semi_inv conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 50, 100]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 4,
                        "lr": 0.001,
                        "classification_weight": 0.1,
                        "pretrained_weights_root": "../TorchSSL/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        "embedding_prop": False,
                        "distance_type": "prototypical",

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
                        "pretrain_method": 'flexmatch',
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['flexmatch_proto_lr0001'] += [new_setting]

EXP_GROUPS["flexmatch_proto_hp"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_inv]:#[, resnet12_semi_inv conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [100]:#[20, 50, 100 20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        "lr": 0.001,
                        "classification_weight": 1,
                        "pretrained_weights_root": "../TorchSSL/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        "embedding_prop": False,
                        "distance_type": "prototypical",

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
                        "pretrain_method": 'flexmatch',
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['flexmatch_proto_hp'] += [new_setting]

EXP_GROUPS["flexmatch_proto_cw05"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_inv]:#[, resnet12_semi_inv conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 50, 100]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 4,
                        # "lr": 0.01,
                        "pretrained_weights_root": "../TorchSSL/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        "embedding_prop": False,
                        "distance_type": "prototypical",

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
                        "pretrain_method": 'flexmatch',
                        "classification_weight": 0.5
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['flexmatch_proto_cw05'] += [new_setting]

EXP_GROUPS["flexmatch_proto_tiered_lr0001"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_tiered_imagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_inv]:#[ resnet12_semi_inv] conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 100, 200]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 4,
                        "lr": 0.001,
                        "pretrained_weights_root": "../TorchSSL/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        "embedding_prop": False,
                        "distance_type": "prototypical",

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
                        "pretrain_method": 'flexmatch',
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['flexmatch_proto_tiered_lr0001'] += [new_setting]


shot=5
query_shot = 15

EXP_GROUPS["finetune_cub_lr0001"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_cub]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test, resnet12_semi_norm]:#[conv4semi_test, resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [2, 6, 10]:#[2, 6, 10]:#[20, 60, 120, 300, 540]:
                for lr in [0.0005]:#[0.001, 0.01, 0.005, 0.0005]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        "lr":lr,
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
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['finetune_cub_lr0001'] += [new_setting]

EXP_GROUPS["proto_finetune_cub_lr0001"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_cub]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [resnet12_semi_norm, conv4semi_test]:#[conv4semi_test, conv4semi_test, resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [2, 6, 10]:#[2, 6, 10]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        "lr":0.0005,#0.001,
                        # "classification_weight": 0.1,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_cub",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        "embedding_prop": False,
                        "distance_type": "prototypical",

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
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['proto_finetune_cub_lr0001'] += [new_setting]

EXP_GROUPS["flexmatch_finetune_cub_lr0001"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_cub]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [ resnet12_semi_inv, conv4semi_inv]:#[ conv4semi_inv,  conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [2, 6, 10]:#[2, 6, 10 20, 60, 120, 300, 540]:
                for lr in [0.001]:#[0.01, 0.005, 0.0005]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        "lr": lr,
                        "pretrained_weights_root": "../TorchSSL/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        # "embedding_prop": False,
                        # "distance_type": "prototypical",

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
                        "pretrain_method": 'flexmatch',
                        "support_size_train": shot,
                        "query_size_train": query_shot,
                        "support_size_val": shot,
                        "query_size_val": 15,
                        "support_size_test": shot,
                        "query_size_test": 15,
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['flexmatch_finetune_cub_lr0001'] += [new_setting]

EXP_GROUPS["flexmatch_proto_cub_lr0001"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_cub]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_inv, resnet12_semi_inv]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [2, 6, 10]:#[2, 6, 10][20, 60, 120, 300, 540]:
                for lr in [0.001]:# [0.005,0.002, 0.0005]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        "lr": lr,
                        "classification_weight": 0.1,
                        "pretrained_weights_root": "../TorchSSL/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        "embedding_prop": False,
                        "distance_type": "prototypical",

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
                        "pretrain_method": 'flexmatch',
                        "support_size_train": shot,
                        "query_size_train": query_shot,
                        "support_size_val": shot,
                        "query_size_val": 15,
                        "support_size_test": shot,
                        "query_size_test": 15,
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['flexmatch_proto_cub_lr0001'] += [new_setting]

EXP_GROUPS["flexmatch_proto_cub_lr0001_ep"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_cub]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_inv, resnet12_semi_inv]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [2, 6, 10]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        "lr": 0.001,
                        "pretrained_weights_root": "../TorchSSL/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        "embedding_prop": True,
                        "distance_type": "prototypical",

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
                        "pretrain_method": 'flexmatch',
                        "support_size_train": shot,
                        "query_size_train": query_shot,
                        "support_size_val": shot,
                        "query_size_val": 15,
                        "support_size_test": shot,
                        "query_size_test": 15,
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['flexmatch_proto_cub_lr0001_ep'] += [new_setting]

EXP_GROUPS["proto_finetune_cub_lr0001_ep"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_cub]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test, resnet12_semi_norm]:#[conv4semi_test, resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [2, 6, 10]:#[2, 6, 10]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        "lr":0.001,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_cub",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        "embedding_prop": True,
                        "distance_type": "prototypical",

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
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['proto_finetune_cub_lr0001_ep'] += [new_setting]

# base pretraining
shot=5
query_shot = 15
EXP_GROUPS["base_finetune"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test, resnet12_semi_norm]:#[, resnet12_semi_inv conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 50, 100]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 4,
                        # "lr": 0.001,
                        "pretrained_weights_root": "./logs_semi_base_norm/pretraining3",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        # "embedding_prop": False,
                        # "distance_type": "prototypical",

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
                        "pretrain_method": 'base',
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['base_finetune'] += [new_setting]

# classification_weight=0
EXP_GROUPS["finetune_cw0"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test, resnet12_semi_norm]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 50, 100]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 4,
                        # "lr": 0.001,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
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
                        "classification_weight": 0,

                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['finetune_cw0'] += [new_setting]


EXP_GROUPS["proto_finetune_cw0"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test,resnet12_semi_norm]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 50, 100]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        # "ngpu": 1,
                        # "lr": 0.001,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
                        # "pretrained_weights_root": "./logs_test/pretraining",
                        "embedding_prop": False,
                        "distance_type": "prototypical",

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
                        "classification_weight": 0,
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['proto_finetune_cw0'] += [new_setting]

EXP_GROUPS["finetune_only_PL"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test, resnet12_semi_norm]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20, 50, 100]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        # "lr": 0.001,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
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
                        "classification_weight": 0,
                        "only_pseudo_labels": True,

                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['finetune_only_PL'] += [new_setting]

EXP_GROUPS["finetune_noise"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4noise]:#[, resnet12_noise, conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
        for npc in [20]:#, 50, 100]:#[20, 60, 120, 300, 540]:
                # for rbf_scale in [0, 0.01, 0.005, 0.05]:#[0.15, 0.3, 0.5]:#[0.1, 0.2, 0.05]:
                # for ep_knn in [0]:#[5, 10, 1, 0]:
            # for resort in [True, False]:
                for query_drop_rate in [0.2, 0.3, 0.5, 0.0, 0.1]:#[0.2]
                # for oqw, te in [[True, True]]:#,[False, True],[True, False], [True, True] ]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        # "lr": 0.001,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
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
                        "classification_weight": 0.1,
                        # "only_pseudo_labels": True,

                        # noise labels
                        "ep_knn":0,
                        "lp_knn":0,
                        # "rbf_scale": rbf_scale,# 0,#0.01,#rbf_scale,
                        # "weight_loss": True,
                        # "only_query_weight": True, #oqw,#False,
                        "trans_encoder": True,#True, #te,#True
                        "logits_consistency_weight":0,
                        "logits_consistency_knn":5,
                        "resort": True,
                        "query_drop_rate": query_drop_rate,
                        "drop_type": "sort",
                        # "soft_label": True,
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['finetune_noise'] += [new_setting]

EXP_GROUPS["finetune_transformer"]=[]
base_setting = EXP_GROUPS['finetune'][0]
for dataset in [semi_miniimagenet]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            for npc in [20]:#, 50, 100]:#[20, 60, 120, 300, 540]:
                # for rbf_scale in [0.2]:#[0.15, 0.3, 0.5]:#[0.1, 0.2, 0.05]:
                for oqw, te in [[True, True]]:#,[False, True],[True, False], [True, True] ]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        # "lr": 0.001,
                        "pretrained_weights_root": "../semco/saved_models/",#"./logs_semi_base_norm/pretrain_tiered",
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
                        "classification_weight": 0.1,
                        "only_pseudo_labels": False,

                        # noise labels
                        # "rbf_scale": 0.2,#rbf_scale,
                        # "weight_loss": True,
                        # "only_query_weight": oqw,#False,
                        "trans_encoder": te,#True
                        "trans_encoder_cls": True,
                        "embedding_prop": True,

                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['finetune_transformer'] += [new_setting]