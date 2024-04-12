import numpy as np
import os
import pickle as pkl
import torchvision.transforms as T
# from torchvision.transforms import GaussianBlur
from utils.semco.randaugment import RandomAugment
import utils.semco.transform as Ts

def load_embs(data_root, label_names, split):
    # load embedding
    embeddings = np.load(
        os.path.join(data_root, 'few-shot-wordemb-{}.npz'.format(split)),
        allow_pickle=True)
    embeddings = embeddings['features']
    embeddings = embeddings.item()
    embs = []
    for i, name in enumerate(sorted(label_names)):
        embs.append(embeddings[name])

    return embs

def load_embs_dict(data_root, label_names, split):
    # load embedding
    embeddings = np.load(
        os.path.join(data_root, 'few-shot-wordemb-{}.npz'.format(split)),
        allow_pickle=True)
    embeddings = embeddings['features']
    embeddings = embeddings.item()
    embs = {}
    sort_names = sorted(label_names)
    for i, name in enumerate(sort_names):
        embs[name] = embeddings[name]

    return embs, sort_names

def load_embs_dict_semco(data_root, label_names, split):
    # load embedding
    embeddings = np.load(
        os.path.join(data_root, 'few-shot-wordemb-{}.npz'.format(split)),
        allow_pickle=True)
    embeddings = embeddings['features']
    embeddings = embeddings.item()
    embs = {}
    sort_names = sorted(label_names)
    for i, name in enumerate(sort_names):
        embs[name] = embeddings[name]

    return embs, sort_names

def load_split_data(data_root, features0, labels0, split):
    with open(data_root % split, 'rb') as infile:
        data_index = pkl.load(infile)
    ind = data_index['supervised_index']
    # split features and labels
    features = features0[ind]
    labels = labels0[ind]
    ind = data_index['unsupervised_index']
    # split features and labels
    un_features = features0[ind]
    un_labels = labels0[ind]
    return features, labels, un_features, un_labels

def load_semi_data(data_root, features0, labels0, split):
    with open(data_root % split, 'rb') as infile:
        data_index = pkl.load(infile)
    # labeled features and labels
    ind = data_index['supervised_index']
    features = features0[ind]
    labels = labels0[ind]
    # unlabeled features and labels
    ind = data_index['unsupervised_index']
    un_features = features0[ind]
    un_labels = labels0[ind]
    # test features and labels
    ind = data_index['test_index']
    test_features = features0[ind]
    test_labels = labels0[ind]
    return features, labels, un_features, un_labels, test_features, test_labels

def load_semi_data_v2(data_root, features0, labels0, split, data_group):
    with open(data_root % split, 'rb') as infile:
        data_index = pkl.load(infile)
    if data_group == 'train_label':
        ind = data_index['supervised_index']
    elif data_group == 'train_unlabel':
        ind = data_index['unsupervised_index']
    elif data_group == 'train_test':
        ind = data_index['test_index']
    elif data_group == 'train_all':
        ind = data_index['supervised_index']
        ind_u = data_index['unsupervised_index']
        ind = np.concatenate([ind,ind_u])
    else:
        raise NotImplementedError
    features = features0[ind]
    labels = labels0[ind]
    return features, labels

# def get_simsiam_trans(image_size):
#     imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
#     mean_std = imagenet_mean_std
#     image_size = 224 if image_size is None else image_size  # by default simsiam use image size 224
#     p_blur = 0.5 if image_size > 32 else 0  # exclude cifar
#     transform = T.Compose([
#         T.transforms.ToPILImage(),
#         T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
#         T.RandomHorizontalFlip(),
#         T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#         T.RandomGrayscale(p=0.2),
#         T.RandomApply([T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=p_blur),
#         T.ToTensor(),
#         # T.Normalize(*mean_std)
#     ])
#     return transform

def get_weak_trans(image_size):
    imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    mean_std = imagenet_mean_std
    transform = T.Compose([
            T.transforms.ToPILImage(),
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    return transform

def get_strong_trans(image_size):
    imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    mean_std = imagenet_mean_std
    transform = T.Compose([
            T.transforms.ToPILImage(),
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.Lambda(lambda x: np.array(x)),
            RandomAugment(2, 10),
            Ts.Normalize(mean_std[0], mean_std[1]),
            Ts.ToTensor(),
            # RandomAugment(2, 10),
            # Ts.ToTensor(),
            # T.Normalize(*mean_std)
        ])
    return transform

# common for both labelled and unlabelled
#             self.trans_weak = T.Compose([
#                 TTV.Resize(self.size),
#                 TTV.Pad(padding=4, padding_mode='reflect'),
#                 TTV.RandomCrop(self.cropsize),
#                 TTV.RandomHorizontalFlip(p=0.5),
#                 TTV.Lambda(lambda x: np.array(x)),
#                 T.Normalize(self.mean, self.std),
#                 T.ToTensor(),
#             ])
#             self.trans_strong = T.Compose([
#                 TTV.Resize(self.size),
#                 TTV.Pad(padding=4, padding_mode='reflect'),
#                 TTV.RandomCrop(self.cropsize),
#                 TTV.RandomHorizontalFlip(p=0.5),
#                 TTV.Lambda(lambda x: np.array(x)),
#                 RandomAugment(2, 10),
#                 T.Normalize(self.mean, self.std),
#                 T.ToTensor(),
#             ])

# def load_unsupervised_data(data_root, features, labels, split):
#     with open(data_root % split, 'rb') as infile:
#         data_index = pkl.load(infile)
#     ind = data_index['unsupervised_index']
#     # split features and labels
#     features = features[ind]
#     labels = labels[ind]
#     return features, labels
