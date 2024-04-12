
# from . import trancos, fish_reg
from torchvision import transforms
import torchvision
import cv2, os
from src import utils as ut
import pandas as pd
import numpy as np

import pandas as pd
import  numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from .episodic_dataset import FewShotSampler
from .episodic_miniimagenet import EpisodicMiniImagenet, EpisodicMiniImagenetPkl, SemiEpisodicMiniImagenetPkl
from .miniimagenet import NonEpisodicMiniImagenet, RotatedNonEpisodicMiniImagenet, RotatedNonEpisodicMiniImagenetPkl, \
    SemiRotationMiniImagenetPkl
from .episodic_tiered_imagenet import EpisodicTieredImagenet
from .tiered_imagenet import RotatedNonEpisodicTieredImagenet
from .cub import RotatedNonEpisodicCUB, NonEpisodicCUB, SemiNonEpisodicCUB
from .episodic_cub import EpisodicCUB, SemiEpisodicCUB
from .episodic_tiered_imagenet import EpisodicTieredImagenet, SemiEpisodicTieredImagenet
from .tiered_imagenet import RotatedNonEpisodicTieredImagenet, NonEpisodicTieredImagenet, SemiNonEpisodicTieredImagenet
import torchvision.transforms as T
from utils.semco.randaugment import RandomAugment
import utils.semco.transform as Ts

def get_dataset(dataset_name, 
                data_root,
                split, 
                transform, 
                classes,
                support_size,
                query_size, 
                unlabeled_size,
                n_iters,
                rotation_labels=[0, 1, 2, 3],
                unsupervised_loss='rotation',
                data_group='train_label',
                n_distract=0,
                aug=False,
                ):

    transform_func = get_transformer(transform, split)
    transform_func_s = get_transformer_strong(transform, split)
    if dataset_name == "rotated_miniimagenet":
        dataset = RotatedNonEpisodicMiniImagenet(data_root,
                                                 split,
                                                 transform_func)

    elif dataset_name == "miniimagenet":
        dataset = NonEpisodicMiniImagenet(data_root,
                                          split,
                                          transform_func)
    elif dataset_name == "episodic_miniimagenet":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicMiniImagenet(data_root=data_root,
                                       split=split, 
                                       sampler=few_shot_sampler,
                                       size=n_iters,
                                       transforms=transform_func,
                                       unsupervised_loss=unsupervised_loss,
                                       rotation_labels = rotation_labels,
                                       aug = aug,
                                       )
    elif dataset_name == "rotated_episodic_miniimagenet_pkl":
        dataset = RotatedNonEpisodicMiniImagenetPkl(data_root=data_root,
                                          split=split,
                                          transforms=transform_func,
                                        unlabeled_size=unlabeled_size,
                                                    unsupervised_loss=unsupervised_loss,
                                                    rotation_labels=rotation_labels)
    elif dataset_name == "semi_rotated_miniimagenet_pkl":
        dataset = SemiRotationMiniImagenetPkl(data_root=data_root,
                                          split=split,
                                          transforms=transform_func,
                                          transforms_strong=transform_func_s,
                                          unlabeled_size=unlabeled_size,
                                          unsupervised_loss=unsupervised_loss,
                                          rotation_labels=rotation_labels,
                                          data_group=data_group,
                                          n_distract=n_distract)
    elif dataset_name == "episodic_miniimagenet_pkl":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicMiniImagenetPkl(data_root=data_root,
                                       split=split, 
                                       sampler=few_shot_sampler,
                                       size=n_iters,
                                       transforms=transform_func)
    elif dataset_name == "semi_episodic_miniimagenet_pkl":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = SemiEpisodicMiniImagenetPkl(data_root=data_root,
                                       split=split,
                                       sampler=few_shot_sampler,
                                       size=n_iters,
                                       transforms=transform_func,
                                       data_group=data_group,
                                       aug=aug)
    elif dataset_name == "cub":
        dataset = NonEpisodicCUB(data_root, split, transform_func)
    elif dataset_name == "rotated_cub":
        dataset = RotatedNonEpisodicCUB(data_root, split, transform_func)
    elif dataset_name == "episodic_cub":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicCUB(data_root=data_root,
                              split=split, 
                              sampler=few_shot_sampler,
                              size=n_iters,
                              transforms=transform_func)
    elif dataset_name == "semi_cub":
        dataset = SemiNonEpisodicCUB(data_root, split, transform_func,
                                     rotation_labels=rotation_labels, data_group=data_group)
    elif dataset_name == "semi_episodic_cub":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = SemiEpisodicCUB(data_root=data_root,  split=split,
                                  sampler=few_shot_sampler, size=n_iters, transforms=transform_func,data_group=data_group)
    elif dataset_name == "tiered-imagenet":
        dataset = NonEpisodicTieredImagenet(data_root, split, transform_func)
    elif dataset_name == "rotated_tiered-imagenet":
        dataset = RotatedNonEpisodicTieredImagenet(data_root, split, transform_func)
    elif dataset_name == "episodic_tiered-imagenet":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicTieredImagenet(data_root,
                                         split=split, 
                                         sampler=few_shot_sampler,
                                         size=n_iters,
                                         transforms=transform_func)
    elif dataset_name == "semi_tiered-imagenet":
        dataset = SemiNonEpisodicTieredImagenet(data_root=data_root,
                                          split=split,
                                          transforms=transform_func,
                                          rotation_labels=rotation_labels,
                                          data_group=data_group)
    elif dataset_name == "semi_episodic_tiered-imagenet":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = SemiEpisodicTieredImagenet(data_root,
                                         split=split,
                                         sampler=few_shot_sampler,
                                         size=n_iters,
                                         transforms=transform_func,
                                         data_group=data_group)

    return dataset



# ===================================================
# helpers
def my_inv_norm(img):
    img = (img + 1.0) / 2.0
    return img

def get_transformer(transform, split):
    imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    if transform == "data_augmentation":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((84,84)),
                                                    torchvision.transforms.ToTensor()])

        return transform
    if transform == 'simsiam':
        imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        mean_std = imagenet_mean_std
        image_size = 84  # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0  # exclude cifar
        transform = T.Compose([
            T.transforms.ToPILImage(),
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            # T.Normalize(*mean_std)
        ])
        return transform

    if transform == 'imagnet_nomalize':
        imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        mean_std = imagenet_mean_std
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((84, 84)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(*mean_std)
                                                    ])
        return transform

    if transform == 'imagnet_nomalize_inv':
        imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        mean_std = imagenet_mean_std
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((84, 84)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(*mean_std),
                                                    torchvision.transforms.Lambda(lambda img: my_inv_norm(img))
                                                    ])
        return transform

    if transform == 'imagnet_nomalize224':
        imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        mean_std = imagenet_mean_std
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((224, 224)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(*mean_std)
                                                    ])
        return transform

    if "{}_{}".format(transform, split) == "cifar_train":
        transform = torchvision.transforms.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return transform

    if "{}_{}".format(transform, split) == "cifar_test" or "{}_{}".format(transform, split) == "cifar_val":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return transform

    if transform == "basic":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((84,84)),
                                                    torchvision.transforms.ToTensor()])

        return transform 

    if transform == "wrn_pretrain_train":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.RandomResizedCrop((80, 80), scale=(0.08, 1)),
                                                    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                    torchvision.transforms.ToTensor()
                                                            ])
        return transform
    elif transform == "wrn_finetune_train":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((92, 92)),
                                                    torchvision.transforms.CenterCrop(80),
                                                    torchvision.transforms.RandomHorizontalFlip(),
                                                    torchvision.transforms.ToTensor()
                                                    ])
        return transform

    elif transform == "wrn_pretrain_train_im_norm":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.RandomResizedCrop((80, 80), scale=(0.08, 1)),
                                                    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(*imagenet_mean_std)
                                                            ])
        return transform
    elif transform == "wrn_finetune_train_im_norm":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((92, 92)),
                                                    torchvision.transforms.CenterCrop(80),
                                                    torchvision.transforms.RandomHorizontalFlip(),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(*imagenet_mean_std)
                                                    ])
        return transform

    elif "wrn_im_norm" in transform:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((92, 92)),
                                                    torchvision.transforms.CenterCrop(80),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(*imagenet_mean_std)])
        return transform

    elif "wrn" in transform:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((92, 92)),
                                                    torchvision.transforms.CenterCrop(80),
                                                    torchvision.transforms.ToTensor()])
        return transform

    raise NotImplementedError
                                                    
def get_transformer_strong(transform, split):
    imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    if transform == "data_augmentation":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((84,84)),
                                                    T.Lambda(lambda x: np.array(x)),
                                                    RandomAugment(2, 10),
                                                    # Ts.Normalize(mean_std[0], mean_std[1]),
                                                    Ts.ToTensor(),
                                                    ])

        return transform
    if transform == 'simsiam':
        imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        mean_std = imagenet_mean_std
        image_size = 84  # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0  # exclude cifar
        transform = T.Compose([
            T.transforms.ToPILImage(),
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            # T.Normalize(*mean_std)
        ])
        return transform

    if transform == 'imagnet_nomalize':
        imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        mean_std = imagenet_mean_std
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((84, 84)),
                                                    T.Lambda(lambda x: np.array(x)),
                                                    RandomAugment(2, 10),
                                                    Ts.Normalize(mean_std[0], mean_std[1]),
                                                    Ts.ToTensor(),
                                                    ])
        return transform

    if transform == 'imagnet_nomalize_inv':
        imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        mean_std = imagenet_mean_std
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((84, 84)),
                                                    T.Lambda(lambda x: np.array(x)),
                                                    RandomAugment(2, 10),
                                                    Ts.Normalize(mean_std[0], mean_std[1]),
                                                    Ts.ToTensor(),
                                                    torchvision.transforms.Lambda(lambda img: my_inv_norm(img))
                                                    ])
        return transform


    if transform == 'imagnet_nomalize224':
        imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        mean_std = imagenet_mean_std
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((224, 224)),
                                                    T.Lambda(lambda x: np.array(x)),
                                                    RandomAugment(2, 10),
                                                    Ts.Normalize(mean_std[0], mean_std[1]),
                                                    Ts.ToTensor(),
                                                    ])
        return transform

    if "{}_{}".format(transform, split) == "cifar_train":
        mean_std = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
        transform = torchvision.transforms.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            T.Lambda(lambda x: np.array(x)),
            RandomAugment(2, 10),
            Ts.Normalize(mean_std[0], mean_std[1]),
            Ts.ToTensor(),
            # torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return transform

    if "{}_{}".format(transform, split) == "cifar_test" or "{}_{}".format(transform, split) == "cifar_val":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return transform

    if transform == "basic":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((84,84)),
                                                    T.Lambda(lambda x: np.array(x)),
                                                    RandomAugment(2, 10),
                                                    # Ts.Normalize(mean_std[0], mean_std[1]),
                                                    Ts.ToTensor(),
                                                    ])

        return transform

    if transform == "wrn_pretrain_train":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.RandomResizedCrop((80, 80), scale=(0.08, 1)),
                                                    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                    T.Lambda(lambda x: np.array(x)),
                                                    RandomAugment(2, 10),
                                                    # Ts.Normalize(mean_std[0], mean_std[1]),
                                                    Ts.ToTensor(),
                                                            ])
        return transform
    elif transform == "wrn_finetune_train":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((92, 92)),
                                                    torchvision.transforms.CenterCrop(80),
                                                    torchvision.transforms.RandomHorizontalFlip(),
                                                    T.Lambda(lambda x: np.array(x)),
                                                    RandomAugment(2, 10),
                                                    # Ts.Normalize(mean_std[0], mean_std[1]),
                                                    Ts.ToTensor(),
                                                    # torchvision.transforms.ToTensor()
                                                    ])
        return transform

    elif transform == "wrn_pretrain_train_im_norm":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.RandomResizedCrop((80, 80), scale=(0.08, 1)),
                                                    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                    T.Lambda(lambda x: np.array(x)),
                                                    RandomAugment(2, 10),
                                                    Ts.Normalize(imagenet_mean_std[0], imagenet_mean_std[1]),
                                                    Ts.ToTensor(),
                                                            ])
        return transform
    elif transform == "wrn_finetune_train_im_norm":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((92, 92)),
                                                    torchvision.transforms.CenterCrop(80),
                                                    torchvision.transforms.RandomHorizontalFlip(),
                                                    T.Lambda(lambda x: np.array(x)),
                                                    RandomAugment(2, 10),
                                                    Ts.Normalize(imagenet_mean_std[0], imagenet_mean_std[1]),
                                                    Ts.ToTensor(),
                                                    # torchvision.transforms.ToTensor()
                                                    ])
        return transform

    elif "wrn" in transform:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((92, 92)),
                                                    torchvision.transforms.CenterCrop(80),
                                                    T.Lambda(lambda x: np.array(x)),
                                                    RandomAugment(2, 10),
                                                    # Ts.Normalize(mean_std[0], mean_std[1]),
                                                    Ts.ToTensor(),
                                                    # torchvision.transforms.ToTensor()
                                                    ])
        return transform

    raise NotImplementedError