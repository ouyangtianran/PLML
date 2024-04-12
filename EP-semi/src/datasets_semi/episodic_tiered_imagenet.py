import sys
import torchvision
import torch
from torch.utils.data import Dataset
from .episodic_dataset import EpisodicDataset, FewShotSampler
import json
import os
import numpy as np
import numpy
import cv2
import pickle as pkl
from src.datasets_semi.utils import load_semi_data, load_semi_data_v2

# Inherit order is important, FewShotDataset constructor is prioritary
class EpisodicTieredImagenet(EpisodicDataset):
    tasks_type = "clss"
    name = "tiered-imagenet"
    split_paths = {"train":"train", "test":"test", "valid": "val"}
    c = 3
    h = 84
    w = 84
    def __init__(self, data_root, split, sampler, size, transforms):
        self.data_root = data_root
        self.split = split
        img_path = os.path.join(self.data_root, "%s_images_png.pkl" %(split))
        label_path = os.path.join(self.data_root, "%s_labels.pkl" %(split))
        with open(img_path, 'rb') as infile:
            self.features = pkl.load(infile, encoding="bytes")
        with open(label_path, 'rb') as infile:
            labels = pkl.load(infile, encoding="bytes")['label_specific']
        super().__init__(labels, sampler, size, transforms)
    
    def sample_images(self, indices):
        return [cv2.imdecode(self.features[i], cv2.IMREAD_COLOR)[:,:,::-1] for i in indices]

    def __iter__(self):
        return super().__iter__()


class SemiEpisodicTieredImagenet(EpisodicDataset):
    tasks_type = "clss"
    name = "tiered-imagenet"
    split_paths = {"train": "train", "test": "test", "valid": "val"}
    c = 3
    h = 84
    w = 84

    def __init__(self, data_root, split, sampler, size, transforms, data_group='train_label'):
        self.data_root = data_root
        t_split = split.split('-')
        if len(t_split) > 1:
            # flag to split the dataset to supervised and unsupervised
            split_semi = split  # such as 'train-K-002'
            split = t_split[0]
            semi_path = os.path.join(self.data_root, "%s_labels.pkl")
        self.split = split
        img_path = os.path.join(self.data_root, "%s_images_png.pkl" % (split))
        label_path = os.path.join(self.data_root, "%s_labels.pkl" % (split))
        with open(img_path, 'rb') as infile:
            features = pkl.load(infile, encoding="bytes")
            features = [cv2.imdecode(f, cv2.IMREAD_COLOR)[:, :, ::-1] for f in features]
        with open(label_path, 'rb') as infile:
            labels = pkl.load(infile, encoding="bytes")['label_specific']

        if len(t_split) > 1:
            self.features, self.labels = load_semi_data_v2(semi_path, np.asarray(features), labels,
                                                           split_semi, data_group)
        # self.labels_weight = np.ones_like(self.labels)
        # self.size = len(self.features)
        self.un_features, self.un_labels = load_semi_data_v2(semi_path, np.asarray(features), labels,
                                                           split_semi, 'train_unlabel')
        labels = self.labels

        super().__init__(labels, sampler, size, transforms, un_labels=self.un_labels)

    def sample_images(self, indices):
        # return [cv2.imdecode(self.features[i], cv2.IMREAD_COLOR)[:, :, ::-1] for i in indices]
        return self.features[indices]

    def __iter__(self):
        return super().__iter__()

if __name__ == '__main__':
    import sys
    from torch.utils.data import DataLoader
    from tools.plot_episode import plot_episode
    sampler = FewShotSampler(5, 5, 15, 0)
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                 torchvision.transforms.Resize((84,84)),
                                                 torchvision.transforms.ToTensor(),
                                                ])
    dataset = EpisodicTieredImagenet("train", sampler, 10, transforms)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)
    for batch in loader:
        plot_episode(batch[0], classes_first=False)
