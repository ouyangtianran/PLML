import numpy as np
import torch
import torchvision
import os
from .episodic_dataset import EpisodicDataset, FewShotSampler
import pickle as pkl
from .utils import load_embs, load_split_data,  load_semi_data, load_embs_dict
#get_simsiam_trans,
# Inherit order is important, FewShotDataset constructor is prioritary
class EpisodicMiniImagenet(EpisodicDataset):
    tasks_type = "clss"
    name = "miniimagenet"
    episodic=True
    split_paths = {"train":"train", "valid":"val", "val":"val", "test": "test"}
    # c = 3
    # h = 84
    # w = 84

    def __init__(self, data_root, split, sampler, size, transforms):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        self.data_root = os.path.join(data_root, "mini-imagenet-%s.npz")
        self.split = split
        data = np.load(self.data_root % self.split_paths[split])
        self.features = data["features"]
        labels = data["targets"]
        self.labels = labels
        del(data)
        super().__init__(labels, sampler, size, transforms)

    def sample_images(self, indices):
        return self.features[indices]

    def __iter__(self):
        return super().__iter__()

# Inherit order is important, FewShotDataset constructor is prioritary
class EpisodicMiniImagenetPkl(EpisodicDataset):
    tasks_type = "clss"
    name = "miniimagenet"
    episodic=True
    split_paths = {"train":"train", "valid":"val", "val":"val", "test": "test"}
    # c = 3
    # h = 84
    # w = 84

    def __init__(self, data_root, split, sampler, size, transforms, unsupervised_loss='rotation'):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        self.unsupervised_loss = unsupervised_loss
        self.data_root = os.path.join(data_root, "mini-imagenet-cache-%s.pkl")
        self.split = split
        t_split = split.split('-')
        if len(t_split) > 1:
            # flag to split the dataset to supervised and unsupervised
            split_semi = split  # such as 'train-K-002'
            split = t_split[0]

        with open(self.data_root % self.split_paths[split], 'rb') as infile:
            data = pkl.load(infile)
        self.features = data["image_data"]
        label_names = data["class_dict"].keys()
        labels = np.zeros((self.features.shape[0],), dtype=int)
        for i, name in enumerate(sorted(label_names)):
            labels[np.array(data['class_dict'][name])] = i
        del(data)

        if len(t_split) > 1:
            # load supervised and unsupervised data
            self.features, labels, self.un_features, self.un_labels = \
                load_split_data(self.data_root, self.features, labels,
                                split_semi)
            self.un_size = len(self.un_features)
            self.unlabel_set = np.arange(len(self.un_features))
        un_transforms = None#get_simsiam_trans(84)

        super().__init__(labels, sampler, size, transforms, unsupervised_loss, un_transforms)

        # load word embedding for each class name
        self.embs = load_embs(data_root, label_names, self.split_paths[split])
        self.embs = np.array(self.embs)

    def sample_images(self, indices):
        return self.features[indices]

    def sample_unlabeled_images(self, num):
        indices = np.random.choice(self.unlabel_set, num, replace=False)
        return self.un_features[indices], self.un_labels[indices]

    def sample_embs(self, indices):
        labels = self.labels[indices]
        embs = self.embs[labels]
        return embs

    def __iter__(self):
        return super().__iter__()

class SemiEpisodicMiniImagenetPkl(EpisodicDataset):
    tasks_type = "clss"
    name = "miniimagenet"
    episodic=True
    split_paths = {"train":"train", "valid":"val", "val":"val", "test": "test"}
    # c = 3
    # h = 84
    # w = 84

    def __init__(self, data_root, split, sampler, size, transforms, unsupervised_loss='rotation', data_group='train_label'):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        self.unsupervised_loss = unsupervised_loss
        self.data_root = os.path.join(data_root, "mini-imagenet-cache-%s.pkl")
        self.split = split
        t_split = split.split('-')
        if len(t_split) > 1:
            # flag to split the dataset to supervised and unsupervised
            split_semi = split  # such as 'train-K-002'
            split = t_split[0]

        with open(self.data_root % self.split_paths[split], 'rb') as infile:
            data = pkl.load(infile)
        self.features = data["image_data"]
        label_names = data["class_dict"].keys()
        labels = np.zeros((self.features.shape[0],), dtype=int)
        for i, name in enumerate(sorted(label_names)):
            labels[np.array(data['class_dict'][name])] = i
        del(data)

        # if len(t_split) > 1:
        #     # load supervised and unsupervised data
        #     self.features, labels, self.un_features, self.un_labels = \
        #         load_split_data(self.data_root, self.features, labels,
        #                         split_semi)
        #     self.un_size = len(self.un_features)
        #     self.unlabel_set = np.arange(len(self.un_features))
        features, labels, un_features, un_labels, test_features, test_labels = load_semi_data(self.data_root,
                                                                                              self.features,
                                                                                              labels,
                                                                                              split_semi)
        if data_group == 'train_label':
            self.features, self.labels = features, labels
        elif data_group == 'train_unlabel':
            self.features, self.labels = un_features, un_labels
        elif data_group == 'train_test':
            self.features, self.labels = test_features, test_labels
        else:
            raise NotImplementedError
        self.un_features, self.un_labels = un_features, un_labels
        labels = self.labels
        un_transforms = None#get_simsiam_trans(84)

        super().__init__(labels, sampler, size, transforms, unsupervised_loss, un_transforms, un_labels)

        # load word embedding for each class name
        self.embs = load_embs(data_root, label_names, self.split_paths[split])
        self.embs = np.array(self.embs)
        # fix match semi-supervise
        self.embs_dict, self.class_names = load_embs_dict(data_root, label_names, self.split_paths[split])

    def sample_images(self, indices):
        return self.features[indices]

    def sample_unlabeled_images(self, indices):
        # indices = np.random.choice(self.unlabel_set, num, replace=False), self.un_labels[indices]
        return self.un_features[indices]

    def sample_embs(self, indices):
        labels = self.labels[indices]
        embs = self.embs[labels]
        return embs

    def __iter__(self):
        return super().__iter__()

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from src.tools.plot_episode import plot_episode
    import time
    sampler = FewShotSampler(5, 5, 15, 0)
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                 torchvision.transforms.ToTensor(),
                                                ])
    dataset = EpisodicMiniImagenetPkl('./miniimagenet', 'train', sampler, 1000, transforms)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)
    for batch in loader:
        print(np.unique(batch[0]["targets"].view(20, 5).numpy()))
        # plot_episode(batch[0], classes_first=False)
        # time.sleep(1)

