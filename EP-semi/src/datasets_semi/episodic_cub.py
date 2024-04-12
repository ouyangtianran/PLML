import sys
import torchvision
import torch
from torch.utils.data import Dataset
from .episodic_dataset import EpisodicDataset, FewShotSampler
import json
import os
import numpy as np
from PIL import Image
import numpy
import pickle as pkl
from src.datasets_semi.utils import load_semi_data_v2

class EpisodicCUB(EpisodicDataset):
    h = 84
    w = 84
    c = 3
    name="CUB"
    task="cls"
    split_paths = {"train":"base", "val":"val", "valid":"val", "test":"novel"}
    def __init__(self, data_root, split, sampler, size, transforms):
        self.data_root = data_root
        # self.split = split
        # with open(os.path.join(self.data_root, "few_shot_lists", "%s.json" %self.split_paths[split]), 'r') as infile:
        #     self.metadata = json.load(infile)
        # labels = np.array(self.metadata['image_labels'])
        self.data_root = os.path.join(data_root, "cub-%s.pkl")
        with open(self.data_root % self.split_paths[split], 'rb') as infile:
            data = pkl.load(infile)
        labels = np.array(data['labels'])
        self.features = np.array(data['features'])
        label_map = {l: i for i, l in enumerate(sorted(np.unique(labels)))}
        labels = np.array([label_map[l] for l in labels])
        super().__init__(labels, sampler, size, transforms)
    
    def sample_images(self, indices):
        # return [np.array(Image.open(self.metadata['image_names'][i]).convert("RGB")) for i in indices]
        return self.features[indices]

    def __iter__(self):
        return super().__iter__()


class SemiEpisodicCUB(EpisodicDataset):
    h = 84
    w = 84
    c = 3
    name = "CUB"
    task = "cls"
    split_paths = {"train": "base", "val": "val", "valid": "val", "test": "novel"}

    def __init__(self, data_root, split, sampler, size, transforms, data_group='train_label'):
        self.data_root = data_root
        t_split = split.split('-')
        if len(t_split) > 1:
            # flag to split the dataset to supervised and unsupervised
            split_semi = split  # such as 'train-K-002'
            split = t_split[0]
            semi_path = os.path.join(data_root, "%s.pkl")
            split_semi = split_semi.replace(split, self.split_paths[split])
        self.data_root = os.path.join(data_root, "cub-%s.pkl")
        with open(self.data_root % self.split_paths[split], 'rb') as infile:
            data = pkl.load(infile)
        self.labels = np.array(data['labels'])
        self.features = np.asarray(data['features'])
        if len(t_split) > 1:
            self.features, self.labels = load_semi_data_v2(semi_path, self.features, self.labels,
                                                           split_semi, data_group)
        labels = self.labels
        # self.split = split
        # with open(os.path.join(self.data_root, "few_shot_lists", "%s.json" % self.split_paths[split]), 'r') as infile:
        #     self.metadata = json.load(infile)
        # labels = np.array(self.metadata['image_labels'])
        label_map = {l: i for i, l in enumerate(sorted(np.unique(labels)))}
        labels = np.array([label_map[l] for l in labels])
        super().__init__(labels, sampler, size, transforms)

    def sample_images(self, indices):
        # return [np.array(Image.open(self.metadata['image_names'][i]).convert("RGB")) for i in indices]
        return self.features[indices]

    def __iter__(self):
        return super().__iter__()

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tools.plot_episode import plot_episode
    sampler = FewShotSampler(5, 5, 15, 0)
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                 torchvision.transforms.Resize((84,84)),
                                                 torchvision.transforms.ToTensor(),
                                                ])
    dataset = EpisodicCUB("train", sampler, 10, transforms)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)
    for batch in loader:
        plot_episode(batch[0], classes_first=False)

