 

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


resnet12 = {
    "name": "finetuning",
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

cub = {
    "dataset": "cub",
    "n_classes": 100,
    "dataset_train": "episodic_cub",
    "dataset_val": "episodic_cub",
    "dataset_test": "episodic_cub",
    "data_root": "CUB_200_2011"
}

EXP_GROUPS = {"finetune": []}
lr=0.001
shot = 1
dropout = 0.7
way=5
query_shot = 1
un_mu = 1
p_thr = 0
p_thr_g = 0
npc=20
delay_iter = -1
freeze = 0
un_mu_bs = [[0,128]]#[[0,128],[1,64],[3,32],[7,16]]#[[0,128]]
for dataset in [semi_miniimagenet]:#[miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test]:#[wrn_semi_norm,resnet12_semi_norm,conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
        # for lr in [0.001]:#[0.01, 0.001]:
        # for dropout in [0.3, 0.5]:
        # for way in [5]:
        # for delay_iter in [-1]:#[5,10,20,30]
        # for freeze in [0]:#[4,3,2,1]:#[4,3,2,1,0]
            for un_mu, bs in un_mu_bs:
            # for p_thr in [ 0.1, 0.5, 0.7, 0.95]:
            # for p_thr_g in [0.1, 0.5, 0.7, 0.95, 0]:
        #     for shot in [1]:  # [1]:#
              for npc in [2,6,10]:#[20, 50, 100, 250]:#[20]:#[20, 60, 120, 300, 540]:
                for train_iters in [100]:  # [100, 600]:
                    for classification_weight in [1.5]:#[0, 0.1, 0.5, 1, 1.5, 2, 3]:  # [0, 0.1, 0.5,1,1.5, 2, 3]:
                            EXP_GROUPS['finetune'] += [{"model": backbone,

                                                        # Hardware
                                                        "ngpu": 1,
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
                                                        "pretrained_weights_root": "./logs_semi_base_semco_low_shot/pretraining",

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
                                                        "unlabeled_size_train": 0,

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
                                                        "use_unlabeled_data": "iterating_optimization",#"no_iteration",#"iterating_optimization", #""
                                                        "delay_iter": delay_iter,
                                                        "semi_batch_size": bs,
                                                        "freeze":freeze,
                                                        "cross_entropy_weight": 1,
                                                        }]
