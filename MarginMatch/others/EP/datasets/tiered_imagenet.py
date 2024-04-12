from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle as pkl
from others.EP.datasets.utils import load_semi_data_validation, get_trans, get_weak_trans, get_strong_trans
import cv2

class SemiTieredImagenet(Dataset):
    tasks_type = "clss"
    name = "tiered-imagenet"
    split_paths = {"train":"train", "test":"test", "valid": "val"}
    c = 3
    h = 84
    w = 84

    def __init__(self, data_root, split, size, cropsize, classes,
                 data_group='train_label', mean=None, std=None, **kwargs):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        self.size = size
        self.cropsize = cropsize
        self.classes = classes
        self.n_classes = len(classes)
        self.label_dict = {k: v for v, k in enumerate(classes)}
        self.inverse_label_dict = {v: k for v, k in enumerate(classes)}
        self.type = data_group
        im_size = 84
        # self.data_root = os.path.join(data_root, "mini-imagenet-cache-%s.pkl")
        # t_split = split.split('-')
        # if len(t_split) > 1:
        #     # flag to split the dataset to supervised and unsupervised
        #     split_semi = split  # such as 'train-K-002'
        #     split = t_split[0]
        #
        # with open(self.data_root % self.split_paths[split], 'rb') as infile:
        #     data = pkl.load(infile)
        # self.features = data["image_data"]
        # label_names = data["class_dict"].keys()
        # self.labels = np.zeros((self.features.shape[0],), dtype=int)
        # for i, name in enumerate(sorted(label_names)):
        #     self.labels[np.array(data['class_dict'][name])] = i
        t_split = split.split('-')
        if len(t_split) > 1:
            # flag to split the dataset to supervised and unsupervised
            split_semi = split  # such as 'train-K-002'
            split = t_split[0]
            semi_path = os.path.join(data_root, "%s_labels.pkl")
        split = self.split_paths[split]
        self.data_root = data_root
        img_path = os.path.join(self.data_root, "%s_images_png.pkl" % (split))
        label_path = os.path.join(self.data_root, "%s_labels.pkl" % (split))

        with open(img_path, 'rb') as infile:
            self.features = pkl.load(infile, encoding="bytes")
        with open(label_path, 'rb') as infile:
            self.labels = pkl.load(infile, encoding="bytes")['label_specific']
        self.features = np.array(self.features)

        if len(t_split) > 1:
            features, labels = load_semi_data_validation(semi_path, self.features, self.labels, split_semi, data_group)
        self.features = features
        self.labels = np.int64(labels)
        mean_std = [mean, std]
        if data_group in ['train_label', 'train_unlabel']:

            # common for both labelled and unlabelled
            self.trans_weak = get_weak_trans(self.size, self.cropsize, mean_std)
            self.trans_strong = get_strong_trans(self.size, self.cropsize, mean_std)
        elif data_group in ['train_test', 'train_validation']:

            self.trans = get_trans(self.size, self.cropsize, mean_std)

        self.size = len(self.features)

    def next_run(self):
        pass

    def __getitem__(self, idx):
        # im = self.features[idx]
        im = cv2.imdecode(self.features[idx], cv2.IMREAD_COLOR)[..., ::-1]
        if self.type == 'train_label':
            lb = self.labels[idx]
            return self.trans_weak(im) * 2 -1, self.trans_strong(im) * 2 -1, lb
        elif self.type == 'train_unlabel':
            return self.trans_weak(im) * 2 -1, self.trans_strong(im) * 2 -1
        elif self.type == 'train_validation':
            lb = self.labels[idx]
            return self.trans(im) * 2 -1, lb
        else:
            return self.trans(im) * 2 -1


    def __len__(self):
        return self.size
