import sys
import torchvision
import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
from PIL import Image
from src.datasets_semi.utils import load_semi_data_v2
import pickle as pkl

class NonEpisodicCUB(Dataset):
    name="CUB"
    task="cls"
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
        self.data_root = data_root
        self.split = {"train":"base", "val":"val", "valid":"val", "test":"novel"}[split]
        with open(os.path.join(self.data_root, "few_shot_lists", "%s.json" %self.split), 'r') as infile:
            self.metadata = json.load(infile)
        self.transforms = transforms
        self.rotation_labels = rotation_labels
        self.labels = np.array(self.metadata['image_labels'])
        label_map = {l: i for i, l in enumerate(sorted(np.unique(self.labels)))}
        self.labels = np.array([label_map[l] for l in self.labels])
        self.size = len(self.metadata["image_labels"])

    def next_run(self):
        pass

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
        image = np.array(Image.open(self.metadata["image_names"][item]).convert("RGB"))
        images = self.transforms(image) * 2 - 1
        return images, int(self.labels[item])

    def __len__(self):
        return len(self.labels)

class RotatedNonEpisodicCUB(NonEpisodicCUB):
    name="CUB"
    task="cls"
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
        image = np.array(Image.open(self.metadata["image_names"][item]).convert("RGB"))
        if np.random.randint(2):
            image = np.fliplr(image)
        image_90 = self.transforms(self.rotate_img(image, 90))
        image_180 = self.transforms(self.rotate_img(image, 180))
        image_270 = self.transforms(self.rotate_img(image, 270))
        images = torch.stack([self.transforms(image), image_90, image_180, image_270]) * 2 - 1
        return images, torch.ones(4, dtype=torch.long)*int(self.labels[item]), torch.LongTensor(self.rotation_labels)

class SemiNonEpisodicCUB(Dataset):
        name = "CUB"
        task = "cls"
        split_paths = {"train": "train", "test": "test", "valid": "val"}
        c = 3
        h = 84
        w = 84

        def __init__(self, data_root, split, transforms, rotation_labels=[0, 1, 2, 3], data_group='train_label', **kwargs):
            """ Constructor

            Args:
                split: data split
                few_shot_sampler: FewShotSampler instance
                task: dataset task (if more than one)
                size: number of tasks to generate (int)
                disjoint: whether to create disjoint splits.
            """
            t_split = split.split('-')
            if len(t_split) > 1:
                # flag to split the dataset to supervised and unsupervised
                split_semi = split  # such as 'train-K-002'
                split = t_split[0]
                semi_path = os.path.join(data_root, "%s.pkl")
            self.data_root = data_root
            self.split = {"train": "base", "val": "val", "valid": "val", "test": "novel"}[split]
            split_semi = split_semi.replace(split, self.split)
            # with open(os.path.join(self.data_root, "few_shot_lists", "%s.json" % self.split), 'r') as infile:
            #     self.metadata = json.load(infile)
            self.data_root = os.path.join(data_root, "cub-%s.pkl")
            with open(self.data_root % self.split, 'rb') as infile:
                data = pkl.load(infile)
            self.transforms = transforms
            self.rotation_labels = rotation_labels
            self.labels = np.array(data['labels'])
            self.features = data['features']
            if len(t_split) > 1:
                self.features, self.labels = load_semi_data_v2(semi_path, np.asarray(self.features), self.labels,
                                                               split_semi, data_group)
            label_map = {l: i for i, l in enumerate(sorted(np.unique(self.labels)))}
            self.labels = np.array([label_map[l] for l in self.labels])
            self.size = len(self.labels)
            self.labels_weight = np.ones_like(self.labels)

        def next_run(self):
            pass

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
            # image = np.array(Image.open(self.features[item]).convert("RGB"))
            image = self.features[item]
            r_labels = []
            r_labels.extend(self.rotation_labels)
            if np.random.randint(2):
                image = np.fliplr(image).copy()
            cat = [self.transforms(image)]
            # emb = torch.Tensor(self.embs[self.labels[item]]).view(-1)
            # emb_list = [emb]
            if len(self.rotation_labels) > 1:
                image_90 = self.transforms(self.rotate_img(image, 90))
                image_180 = self.transforms(self.rotate_img(image, 180))
                image_270 = self.transforms(self.rotate_img(image, 270))
                cat.extend([image_90, image_180, image_270])
                # emb_list.extend([emb, emb, emb])

            images = torch.stack(cat) * 2 - 1
            # images = self.transforms(image) * 2 - 1
            # return images, int(self.labels[item]), torch.LongTensor(r_labels), torch.Tensor(image)

            return images, torch.ones(len(r_labels), dtype=torch.long) * int(self.labels[item]), \
                   torch.LongTensor(r_labels), torch.Tensor(image.copy())
            # images = self.transforms(image) * 2 - 1
            # return images, int(self.labels[item])

        def __len__(self):
            return len(self.labels)
