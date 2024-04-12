import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import collections
from copy import deepcopy
import math
from .utils import load_embs, load_split_data#, get_simsiam_trans

_DataLoader = DataLoader
class EpisodicDataLoader(_DataLoader):
    def __iter__(self):
        if isinstance(self.dataset, EpisodicDataset):
            self.dataset.__iter__() 
        else:
            pass    
        return super().__iter__()
torch.utils.data.DataLoader = EpisodicDataLoader

class FewShotSampler():
    FewShotTask = collections.namedtuple("FewShotTask", ["nclasses", "support_size", "query_size", "unlabeled_size"])
    def __init__(self, nclasses, support_size, query_size, unlabeled_size):
        self.task = self.FewShotTask(nclasses, support_size, query_size, unlabeled_size)

    def sample(self):
        return deepcopy(self.task)

class EpisodicDataset(Dataset):
    def __init__(self, labels, sampler, size, transforms, unsupervised_loss='rotation', un_transforms=None, un_labels=None):
        self.unsupervised_loss = unsupervised_loss
        self.labels = labels
        self.labels_weight = np.ones_like(labels)
        self.sampler = sampler
        self.labelset = np.unique(labels)
        self.indices = np.arange(len(labels))
        self.transforms = transforms
        self.size = size
        self.un_transforms = un_transforms
        self.un_labels = un_labels
        if self.un_labels is not None:
            self.un_indices = np.arange(len(un_labels))
            self.un_labels_weight = np.ones_like(un_labels)
        self.reshuffle()

    def reshuffle(self):
        """
        Helper method to randomize tasks again
        """
        self.clss_idx = [np.random.permutation(self.indices[self.labels == label]) for label in self.labelset]
        self.starts = np.zeros(len(self.clss_idx), dtype=int)
        self.lengths = np.array([len(x) for x in self.clss_idx])
        if self.un_labels is not None:
            self.un_clss_idx = [np.random.permutation(self.un_indices[self.un_labels == label]) for label in self.labelset]
            self.un_starts = np.zeros(len(self.un_clss_idx), dtype=int)
            self.un_lengths = np.array([len(x) for x in self.un_clss_idx])

    def gen_few_shot_task(self, nclasses, size, unlabel_size):
        """ Iterates through the dataset sampling tasks

        Args:
            n: FewShotTask.n
            sample_size: FewShotTask.k
            query_size: FewShotTask.k (default), else query_set_size // FewShotTask.n

        Returns: Sampled task or None in the case the dataset has been exhausted.

        """
        classes = np.random.choice(self.labelset, nclasses, replace=False)
        starts = self.starts[classes]
        reminders = self.lengths[classes] - starts
        if np.min(reminders) < size:
            return None, None
        sample_indices = np.array(
            [self.clss_idx[classes[i]][starts[i]:(starts[i] + size)] for i in range(len(classes))])
        sample_indices = np.reshape(sample_indices, [nclasses, size]).transpose()
        self.starts[classes] += size
        if self.un_labels is not None:
            starts = self.un_starts[classes]
            reminders = self.un_lengths[classes] - starts
            if np.min(reminders) < unlabel_size:
                return None, None
            sample_un_indices = np.array(
                [self.un_clss_idx[classes[i]][starts[i]:(starts[i] + unlabel_size)] for i in range(len(classes))])
            sample_un_indices = np.reshape(sample_un_indices, [nclasses, unlabel_size]).transpose()
            self.starts[classes] += unlabel_size
            sample_un_indices = sample_un_indices.flatten()
        else:
            sample_un_indices = None
        return sample_indices.flatten(), sample_un_indices

    def sample_task_list(self):
        """ Generates a list of tasks (until the dataset is exhausted)

        Returns: the list of tasks [(FewShotTask object, task_indices), ...]

        """
        task_list = []
        task_info = self.sampler.sample()
        nclasses, support_size, query_size, unlabeled_size = task_info
        # unlabeled_size = min(unlabeled_size, self.lengths.min() - support_size - query_size)
        task_info = FewShotSampler.FewShotTask(nclasses=nclasses,
                                                support_size=support_size, 
                                                query_size=query_size, 
                                                unlabeled_size=unlabeled_size)
        k = support_size + query_size - unlabeled_size
        if np.any(k > self.lengths):
            raise RuntimeError("Requested more samples than existing")
        few_shot_task, few_shot_task_un = self.gen_few_shot_task(nclasses, k, unlabeled_size)

        while (few_shot_task is not None and unlabeled_size==0) or (unlabeled_size > 0 and few_shot_task_un is not None):
            task_list.append((task_info, few_shot_task, few_shot_task_un))
            task_info = self.sampler.sample()
            nclasses, support_size, query_size, unlabeled_size = task_info
            k = support_size + query_size - unlabeled_size
            few_shot_task, few_shot_task_un = self.gen_few_shot_task(nclasses, k, unlabeled_size)
        return task_list

    def sample_images(self, indices):
        raise NotImplementedError

    def sample_embs(self, indices):
        raise NotImplementedError

    def sample_unlabeled_images(self, indices):
        raise NotImplementedError

    def __getitem__(self, idx):
        """ Reads the idx th task (episode) from disk

        Args:
            idx: task index

        Returns: task dictionary with (dataset (char), task (char), dim (tuple), episode (Tensor))

        """
        fs_task_info, indices, indices_un = self.task_list[idx]
        ordered_argindices = np.argsort(indices)
        ordered_indices = np.sort(indices)
        nclasses, support_size, query_size, unlabeled_size = fs_task_info
        k = support_size + query_size - unlabeled_size
        _images = self.sample_images(ordered_indices)
        images = torch.stack([self.transforms(_images[i]) for i in np.argsort(ordered_argindices)])
        _embs = self.sample_embs(ordered_indices)
        _embs = torch.Tensor(_embs)
        embs = torch.stack([_embs[i] for i in np.argsort(ordered_argindices)])
        total, c, h, w = images.size()
        assert(total == (k * nclasses))
        images = images.view(k, nclasses, c, h, w)
        embs = embs.view(k, nclasses, -1)
        # embs = embs[:,0,:].view(nclasses,-1)
        # del(_images)
        images = images * 2 - 1
        targets = np.zeros([nclasses * k], dtype=int)
        targets[ordered_argindices] = self.labels[ordered_indices, ...].ravel()
        targets_weight = np.zeros([nclasses * k], dtype=int)
        targets_weight[ordered_argindices] = self.labels_weight[ordered_indices, ...].ravel()
        # randomly get unlabeled samples
        if unlabeled_size > 0:
            ordered_argindices = np.argsort(indices_un)
            ordered_indices = np.sort(indices_un)
            _un_images = self.sample_unlabeled_images(ordered_indices)
            un_images = torch.stack([self.transforms(_un_images[i]) for i in np.argsort(ordered_argindices)])
            un_images = un_images.view(unlabeled_size, nclasses, c, h, w)
            un_images = un_images * 2 - 1
            # concat images and un_images
            k_all = support_size + query_size
            s0 = math.ceil(support_size * k / k_all)
            s0_un = support_size - s0
            images_list = []
            images_list.append(images[:s0, ...])
            images_list.append(un_images[:s0_un, ...])
            images_list.append(images[s0:, ...])
            images_list.append(un_images[s0_un:, ...])
            images = torch.cat(images_list)
            un_targets = np.zeros([nclasses * unlabeled_size], dtype=int)
            un_targets[ordered_argindices] = self.un_labels[ordered_indices, ...].ravel()
            targets = np.concatenate([targets, un_targets])
            un_targets_weight = np.zeros([nclasses * unlabeled_size], dtype=int)
            un_targets_weight[ordered_argindices] = self.un_labels_weight[ordered_indices, ...].ravel()
            targets_weight = np.concatenate([targets_weight, un_targets_weight])
            # if self.unsupervised_loss == 'simsiam':
            #     un_images1 = torch.stack([self.transforms(_un_images[i]) for i in range(len(_un_images))])
            #     un_images = torch.stack([un_images, un_images1])
        sample = {"dataset": self.name,
                  "channels": c,
                  "height": h,
                  "width": w,
                  "nclasses": nclasses,
                  "support_size": support_size,
                  "query_size": query_size,
                  "unlabeled_size": unlabeled_size,
                  "targets": torch.from_numpy(targets),
                  "targets_weight": torch.from_numpy(targets_weight),
                  "support_set": images[:support_size, ...],
                  "query_set": images[support_size:(support_size +
                                                   query_size), ...],
                  "embs": embs,
                  "unlabeled_set": None if unlabeled_size == 0 else un_images,
                  "un_labels": None if unlabeled_size == 0 else un_targets}
        return sample    


    def __iter__(self):
        # print("Prefetching new epoch episodes")
        self.task_list = []
        while len(self.task_list) < self.size:
            self.reshuffle()
            self.task_list += self.sample_task_list()
        # print("done prefetching.")
        return []

    def __len__(self):
        return self.size