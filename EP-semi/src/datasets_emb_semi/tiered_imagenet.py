import os
import cv2
import PIL
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset
import torch
from src.datasets_emb_semi.utils import load_embs, load_split_data, \
    get_weak_trans, get_strong_trans, load_embs_dict, load_semi_data

class NonEpisodicTieredImagenet(Dataset):
    tasks_type = "clss"
    name = "tiered-imagenet"
    split_paths = {"train":"train", "test":"test", "valid": "val"}
    c = 3
    h = 84
    w = 84

    def __init__(self, data_root, split, transforms, rotation_labels=[0, 1, 2, 3], **kwargs):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        split = self.split_paths[split]
        self.data_root = data_root
        img_path = os.path.join(self.data_root, "%s_images_png.pkl" %(split))
        label_path = os.path.join(self.data_root, "%s_labels.pkl" %(split))
        self.transforms = transforms
        self.rotation_labels = rotation_labels
        with open(img_path, 'rb') as infile:
            self.features = pkl.load(infile, encoding="bytes")
        with open(label_path, 'rb') as infile:
            self.labels = pkl.load(infile, encoding="bytes")[b'label_specific']

    def next_run(self):
        pass

    def __getitem__(self, item):
        image = cv2.imdecode(self.features[item], cv2.IMREAD_COLOR)[..., ::-1]
        images = self.transforms(image) * 2 - 1
        return images, int(self.labels[item])

    def __len__(self):
        return len(self.labels)

class RotatedNonEpisodicTieredImagenet(NonEpisodicTieredImagenet):
    tasks_type = "clss"
    name = "tiered-imagenet"
    split_paths = {"train":"train", "test":"test", "valid": "val"}
    c = 3
    h = 84
    w = 84

    def __init__(self, *args, **kwargs):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        super().__init__(*args, **kwargs)

    def rotate_img(self, img, rot):
        if rot == 0:  # 0 degrees rotation
            return img
        elif rot == 90:  # 90 degrees rotation
            return np.flipud(np.transpose(img, (1, 0, 2)))
        elif rot == 180:  # 90 degrees rotation
            return np.fliplr(np.flipud(img))
        elif rot == 270:  # 270 degrees rotation / or -90
            return np.transpose(np.flipud(img), (1, 0, 2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

    def __getitem__(self, item):
        image = cv2.imdecode(self.features[item], cv2.IMREAD_COLOR)[..., ::-1]
        if np.random.randint(2):
            image = np.fliplr(image)
        image_90 = self.transforms(self.rotate_img(image, 90))
        image_180 = self.transforms(self.rotate_img(image, 180))
        image_270 = self.transforms(self.rotate_img(image, 270))
        images = torch.stack([self.transforms(image), image_90, image_180, image_270]) * 2 - 1
        return images, torch.ones(4, dtype=torch.long)*int(self.labels[item]), torch.LongTensor(self.rotation_labels)

class SemiNonEpisodicTieredImagenet(Dataset):
    tasks_type = "clss"
    name = "tiered-imagenet"
    split_paths = {"train":"train", "test":"test", "valid": "val"}
    c = 3
    h = 84
    w = 84

    def __init__(self, data_root, split, transforms, rotation_labels=[0, 1, 2, 3], data_group='train_label',**kwargs):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        split = self.split_paths[split]
        t_split = split.split('-')
        if len(t_split) > 1:
            # flag to split the dataset to supervised and unsupervised
            split_semi = split  # such as 'train-K-002'
            split = t_split[0]
        self.data_root = data_root
        img_path = os.path.join(self.data_root, "%s_images_png.pkl" %(split))
        label_path = os.path.join(self.data_root, "%s_labels.pkl" %(split))
        self.transforms = transforms
        self.rotation_labels = rotation_labels
        with open(img_path, 'rb') as infile:
            self.features = pkl.load(infile, encoding="bytes")
        with open(label_path, 'rb') as infile:
            self.labels = pkl.load(infile, encoding="bytes")['label_specific']

        if len(t_split) > 1:
            features, labels, un_features, un_labels, test_features, test_labels = load_semi_data(self.data_root,
                                                                                                  self.features,
                                                                                                  self.labels,
                                                                                                  split_semi)
            if data_group == 'train_label':
                self.features, self.labels = features, labels
            elif data_group == 'train_unlabel':
                self.features, self.labels = un_features, un_labels
            elif data_group == 'train_test':
                self.features, self.labels = test_features, test_labels
            else:
                raise NotImplementedError

    def next_run(self):
        pass

    def __getitem__(self, item):
        image = cv2.imdecode(self.features[item], cv2.IMREAD_COLOR)[..., ::-1]
        images = self.transforms(image) * 2 - 1
        return images, int(self.labels[item])

    def __len__(self):
        return len(self.labels)