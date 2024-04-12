import numpy as np
import torch
from torch.utils.data import Dataset
import datasets.transform as T
from PIL import Image
# from .randaugment import RandomAugment
from datasets.sampler import RandomSampler, BatchSampler
import torchvision.transforms as TTV
from tqdm import tqdm
from pathlib import Path
import os
from others.EP.datasets.miniimagenet import SemiMiniImagenetPkl
from others.EP.datasets.tiered_imagenet import SemiTieredImagenet
from others.EP.datasets.cub import SemiCubPkl

def get_train_loaders(dataset_path, dataset_name, split, classes, batch_size, mu,
                      n_iters_per_epoch, size, cropsize, mean=None, std=None, num_workers=2,pin_memory=True):

    if dataset_name == 'mini-imagenet':
        ds_x = SemiMiniImagenetPkl(dataset_path, split, size, cropsize, classes,
                 data_group='train_label', mean=mean, std=std)
        ds_u = SemiMiniImagenetPkl(dataset_path, split, size, cropsize, classes,
                                   data_group='train_unlabel', mean=mean, std=std)
    elif dataset_name == 'tiered-imagenet':
        ds_x = SemiTieredImagenet(dataset_path, split, size, cropsize, classes,
                                   data_group='train_label', mean=mean, std=std)
        ds_u = SemiTieredImagenet(dataset_path, split, size, cropsize, classes,
                                   data_group='train_unlabel', mean=mean, std=std)
    elif dataset_name == 'CUB_200_2011':
        ds_x = SemiCubPkl(dataset_path, split, size, cropsize, classes,
                                   data_group='train_label', mean=mean, std=std)
        ds_u = SemiCubPkl(dataset_path, split, size, cropsize, classes,
                                   data_group='train_unlabel', mean=mean, std=std)

    sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
    batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
    dl_x = torch.utils.data.DataLoader(ds_x, batch_sampler=batch_sampler_x, num_workers=num_workers,
                                       pin_memory=pin_memory)

    sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
    batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
    dl_u = torch.utils.data.DataLoader(ds_u, batch_sampler=batch_sampler_u, num_workers=num_workers,
                                       pin_memory=True)
    return dl_x, dl_u


def get_val_loader(dataset_path, dataset_name, split, classes, batch_size, num_workers, size, cropsize, mean, std, pin_memory=True,cache_imgs=False):
    if dataset_name == 'mini-imagenet':
        ds = SemiMiniImagenetPkl(dataset_path, split, size, cropsize, classes,
                 data_group='train_validation', mean=mean, std=std)
    elif dataset_name == 'tiered-imagenet':
        ds = SemiTieredImagenet(dataset_path, split, size, cropsize, classes,
                 data_group='train_validation', mean=mean, std=std)
    elif dataset_name == 'CUB_200_2011':
        ds = SemiCubPkl(dataset_path, split, size, cropsize, classes,
                 data_group='train_validation', mean=mean, std=std)
    dl = torch.utils.data.DataLoader(ds,shuffle=False,batch_size=batch_size,drop_last=False,num_workers=num_workers,pin_memory=pin_memory)
    return dl

def get_test_loader(dataset_path, dataset_name, split, classes,  batch_size, num_workers, size, cropsize, mean, std, pin_memory=True, cache_imgs=False):
    if dataset_name == 'mini-imagenet':
        ds = SemiMiniImagenetPkl(dataset_path, split, size, cropsize, classes,
                 data_group='train_test', mean=mean, std=std)
    elif dataset_name == 'tiered-imagenet':
        ds = SemiTieredImagenet(dataset_path, split, size, cropsize, classes,
                 data_group='train_test', mean=mean, std=std)
    elif dataset_name == 'CUB_200_2011':
        ds = SemiCubPkl(dataset_path, split, size, cropsize, classes,
                 data_group='train_test', mean=mean, std=std)
    dl = torch.utils.data.DataLoader(ds,shuffle=False,batch_size=batch_size,drop_last=False,num_workers=num_workers,pin_memory=pin_memory)
    return dl