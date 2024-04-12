from haven import haven_utils as hu

conv4 = {
    "name": "pretraining",
    'backbone':'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

conv4s = {
    "name": "pretraining",
    'backbone':'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "simsiam",
    "transform_val": "basic",
    "transform_test": "basic"
}

conv4semi = {
    "name": "pretraining",
    'backbone': 'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "imagnet_nomalize",
    "transform_val": "imagnet_nomalize",
    "transform_test": "imagnet_nomalize"
}

conv4semi_test = {
    "name": "semi_pretraining",
    'backbone': 'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "imagnet_nomalize",
    "transform_val": "imagnet_nomalize",
    "transform_test": "imagnet_nomalize"
}

conv4semi_test_basic = {
    "name": "semi_pretraining",
    'backbone': 'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

wrn = {
    "name": "pretraining",
    "backbone": 'wrn',
    "depth": 28,
    "width": 10,
    "transform_train": "wrn_pretrain_train",
    "transform_val": "wrn_val",
    "transform_test": "wrn_val"
}

wrn_semi = {
    "name": "semi_pretraining",
    "backbone": 'wrn',
    "depth": 28,
    "width": 10,
    "transform_train": "wrn_pretrain_train",
    "transform_val": "wrn_val",
    "transform_test": "wrn_val"
}

wrn_semi_norm = {
    "name": "semi_pretraining",
    "backbone": 'wrn',
    "depth": 28,
    "width": 10,
    "transform_train": "wrn_pretrain_train_im_norm",
    "transform_val": "wrn_im_norm_val",
    "transform_test": "wrn_im_norm_val"
}


resnet12 = {
    "name": "pretraining",
    "backbone": 'resnet12',
    "depth": 12,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

resnet12_semi = {
    "name": "semi_pretraining",
    "backbone": 'resnet12',
    "depth": 12,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

resnet12_semi_norm = {
    "name": "semi_pretraining",
    "backbone": 'resnet12',
    "depth": 12,
    "width": 1,
    "transform_train": "imagnet_nomalize",
    "transform_val": "imagnet_nomalize",
    "transform_test": "imagnet_nomalize"
}

miniimagenet = {
    "dataset": "miniimagenet",
    "dataset_train": "rotated_episodic_miniimagenet_pkl",
    "dataset_val": "episodic_miniimagenet_pkl",
    "dataset_test": "episodic_miniimagenet_pkl",
    "n_classes": 64,
    "data_root": "mini-imagenet"
}

semi_miniimagenet = {
    "dataset": "semi_miniimagenet",
    "dataset_train": "semi_rotated_miniimagenet_pkl",
    "dataset_train_test": "semi_rotated_miniimagenet_pkl",
    "dataset_val": "episodic_miniimagenet_pkl",
    "dataset_test": "episodic_miniimagenet_pkl",
    "n_classes": 64,
    "data_root": "mini-imagenet"
}

tiered_imagenet = {
    "dataset": "tiered-imagenet",
    "n_classes": 351,
    "dataset_train": "rotated_tiered-imagenet",
    "dataset_val": "episodic_tiered-imagenet",
    "dataset_test": "episodic_tiered-imagenet",
    "data_root": "tiered-imagenet",
}

cub = {
    "dataset": "cub",
    "n_classes": 100,
    "dataset_train": "rotated_cub",
    "dataset_val": "episodic_cub",
    "dataset_test": "episodic_cub",
    "data_root": "CUB_200_2011"
}

EXP_GROUPS = {"pretrain": []}

lr = 0.01
wew = 2 # 2 for test,  1 for 4 gpu training
npc = 20
T=1
trans = ""#"normalize"
un_mu = 3
un_mu_bs = [[3,32]]#[[1,64],[3,32],[7,16]]
shot=1
query_shot=1
ul = "semco" #"semco",#["simsiam","word_proto","rotation","classifier_loss"]
rotation_labels = [0, 1, 2, 3]#[0, 1, 2, 3]
rotation_weight = 1
embedding_prop = True#True,
distance_type = "labelprop"# "labelprop",#"prototypical"
for dataset in [semi_miniimagenet]:#[miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4semi_test]:#[resnet12_semi_norm,wrn_semi_norm,conv4semi_test,conv4semi_test_basic, conv4semi_test,conv4, resnet12, wrn]:
        # for lr in [0.01]:#[0.2, 0.1, 0.05, 0.02,0.01,0.005]: #[0.2, 0.1]:
        # for wew in [0.5, 10]:
        for npc in [2, 6, 10]:#[20, 50, 100, 250]:
        # for T in [0.1, 0.2, 2, 10, 20, 50, 100]:
        # for trans in ["normalize"]:#"Isomap", "Spectral", "MDS",
        # for un_mu in [0]:#[0,1,2,3]
            for un_mu, bs in un_mu_bs:
                EXP_GROUPS['pretrain'] += [{"model": backbone,

                                        # Hardware
                                        "ngpu": 1,
                                        "random_seed": 42,

                                        # Optimization
                                        "batch_size": bs,
                                        "target_loss": "val_accuracy",#"train_test_accuracy",#
                                        "lr": lr,
                                        "min_lr_decay": 0.0001,
                                        "weight_decay": 0.0005,
                                        "patience": 10,
                                        "max_epoch": 200,
                                        "train_iters": 600,
                                        "val_iters": 600,
                                        "test_iters": 600,
                                        "tasks_per_batch": 1,

                                        # Model
                                        "dropout": 0.1,
                                        "avgpool": True,

                                        # Data
                                        'n_classes': dataset["n_classes"],
                                        "collate_fn": "default",
                                        "transform_train": backbone["transform_train"],
                                        "transform_val": backbone["transform_val"],
                                        "transform_test": backbone["transform_test"],

                                        "dataset_train": dataset["dataset_train"],
                                        "dataset_train_test": dataset["dataset_train_test"],
                                        "dataset_train_root": dataset["data_root"],
                                        "classes_train": 5,
                                        "support_size_train": shot,
                                        "query_size_train": query_shot,
                                        "unlabeled_size_train": 0,

                                        "dataset_val": dataset["dataset_val"],
                                        "dataset_val_root": dataset["data_root"],
                                        "classes_val": 5,
                                        "support_size_val": shot,
                                        "query_size_val": 15,
                                        "unlabeled_size_val": 0,

                                        "dataset_test": dataset["dataset_test"],
                                        "dataset_test_root": dataset["data_root"],
                                        "classes_test": 5,
                                        "support_size_test": shot,
                                        "query_size_test": 15,
                                        "unlabeled_size_test": 0,
                                        

                                        # Hparams
                                        "embedding_prop": embedding_prop,
                                        "cross_entropy_weight": 1,
                                        "few_shot_weight": 0,
                                        "rotation_weight": rotation_weight,
                                        "active_size": 0,
                                        "distance_type": distance_type,#"labelprop",#"prototypical"
                                        "kernel_bound": "",
                                        "rotation_labels": rotation_labels,#[0, 1, 2, 3],#[0],#

                                        # word embedding
                                        "word_emb_weight": 0,
                                        "adjust_type": "linear",
                                        "adjust_dropout": 0.7,
                                        "word_emb_trans":trans,#"Isomap", "Spectral", "MDS",
                                        "word_emb_loss": "l2",
                                        "pretrained_weights_root":None,#"./logs_test/pretraining",None

                                        # semi dataset
                                        "num_per_class": npc,
                                        "unlabeled_multiple": un_mu, # the multiple of labeled data in each batch, such as label:unlabel, 1:1, when unlabeled_multiple=1
                                        "temperature": T,
                                        "unsupervised_loss": ul,#"semco",#["simsiam","word_proto","rotation","classifier_loss"]
                                        "lambda_emb": 0,
                                        "unsupervised_weight": 1,
                                        "semi_batch_size": bs,
                                        "remove_noise": "half",#["no","half","keep_visual"]
                                        }]
