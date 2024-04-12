from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle as pkl
from src.datasets_semi.utils import load_embs, load_split_data, \
    get_weak_trans, get_strong_trans, load_embs_dict, load_semi_data, load_semi_data_v2#, get_simsiam_trans

class NonEpisodicMiniImagenet(Dataset):
    tasks_type = "clss"
    name = "miniimagenet"
    split_paths = {"train": "train", "val":"val", "valid": "val", "test": "test"
                   }
    episodic=False
    c = 3
    h = 84
    w = 84

    def __init__(self, data_root, split, transforms, **kwargs):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        self.data_root = os.path.join(data_root, "mini-imagenet-%s.npz")
        data = np.load(self.data_root % self.split_paths[split])
        self.features = data["features"]
        self.labels = data["targets"]
        self.transforms = transforms



    def next_run(self):
        pass

    def __getitem__(self, item):
        image = self.transforms(self.features[item])
        image = image * 2 - 1
        return image, self.labels[item]

    def __len__(self):
        return len(self.features)

class RotatedNonEpisodicMiniImagenet(Dataset):
    tasks_type = "clss"
    name = "miniimagenet"
    split_paths = {"train": "train", "val":"val", "valid": "val", "test": "test"}
    episodic=False
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
        self.data_root = os.path.join(data_root, "mini-imagenet-%s.npz")
        data = np.load(self.data_root % self.split_paths[split])
        self.features = data["features"]
        self.labels = data["targets"]
        self.transforms = transforms
        self.size = len(self.features)
        self.rotation_labels = rotation_labels

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
        image = self.features[item]
        if np.random.randint(2):
            image = np.fliplr(image).copy()
        cat = [self.transforms(image)]
        if len(self.rotation_labels) > 1:
            image_90 = self.transforms(self.rotate_img(image, 90))
            image_180 = self.transforms(self.rotate_img(image, 180))
            image_270 = self.transforms(self.rotate_img(image, 270))
            cat.extend([image_90, image_180, image_270])
        images = torch.stack(cat) * 2 - 1
        return images, torch.ones(len(self.rotation_labels), dtype=torch.long)*int(self.labels[item]), torch.LongTensor(self.rotation_labels)

    def __len__(self):
        return self.size
    
class RotatedNonEpisodicMiniImagenetPkl(Dataset):
    tasks_type = "clss"
    name = "miniimagenet"
    split_paths = {"train": "train", "val":"val", "valid": "val", "test": "test"}
    episodic=False
    c = 3
    h = 84
    w = 84

    def __init__(self, data_root, split, transforms, rotation_labels=[0, 1, 2, 3],
                 unlabeled_size=0, unsupervised_loss='rotation', **kwargs):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        im_size = 84
        self.unsupervised_loss = unsupervised_loss
        self.unlabeled_size = unlabeled_size
        self.un_transforms = get_simsiam_trans(im_size)
        self.data_root = os.path.join(data_root, "mini-imagenet-cache-%s.pkl")
        t_split = split.split('-')
        if len(t_split) > 1:
            # flag to split the dataset to supervised and unsupervised
            split_semi = split # such as 'train-K-002'
            split = t_split[0]

        with open(self.data_root % self.split_paths[split], 'rb') as infile:
            data = pkl.load(infile)
        self.features = data["image_data"]
        label_names = data["class_dict"].keys()
        self.labels = np.zeros((self.features.shape[0],), dtype=int)
        for i, name in enumerate(sorted(label_names)):
            self.labels[np.array(data['class_dict'][name])] = i

        if len(t_split) > 1:
            # load supervised and unsupervised data
            self.features, self.labels, self.un_features, self.un_labels = \
                load_split_data(self.data_root, self.features, self.labels,
                                split_semi)

        del(data)
        self.transforms = transforms
        self.size = len(self.features)
        self.un_size = len(self.un_features)
        self.rotation_labels = rotation_labels
        # load word embedding for each class name
        self.embs = load_embs(data_root, label_names, self.split_paths[split])
        self.embs = np.array(self.embs)
        # fix match semi-supervise
        self.embs_dict, self.class_names = load_embs_dict(data_root, label_names, self.split_paths[split])
        if unsupervised_loss == 'semco':
            self.transforms = get_weak_trans(im_size)
            self.strong_transforms = get_strong_trans(im_size)


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
        image = self.features[item]
        r_labels = []
        r_labels.extend(self.rotation_labels)
        if np.random.randint(2):
            image = np.fliplr(image).copy()
        cat = [self.transforms(image)]
        emb = torch.Tensor(self.embs[self.labels[item]]).view(-1)
        emb_list = [emb]
        if len(self.rotation_labels) > 1:
            image_90 = self.transforms(self.rotate_img(image, 90))
            image_180 = self.transforms(self.rotate_img(image, 180))
            image_270 = self.transforms(self.rotate_img(image, 270))
            cat.extend([image_90, image_180, image_270])
            emb_list.extend([emb, emb, emb])

        if self.unsupervised_loss == 'semco':
            image_strong = self.strong_transforms(image)
            cat.extend([image_strong])
            emb_list.extend([emb])
            r_labels.extend([0])

        images = torch.stack(cat) * 2 - 1
        embs = torch.stack(emb_list)
        # get unlabeled data
        if self.unlabeled_size > 0:
            cat = []
            un_r_labels = []
            for i in range(self.unlabeled_size):
                un_item = (item + np.random.randint(self.un_size)) % self.un_size
                un_image = self.un_features[un_item]
                if self.unsupervised_loss == 'simsiam':
                    # simsiam transforms (two view)
                    if i == 0:
                        cat.extend([self.un_transforms(image)])
                        cat.extend([self.un_transforms(image)])
                    cat.extend([self.un_transforms(un_image)])
                    cat.extend([self.un_transforms(un_image)])
                    un_r_labels = self.rotation_labels
                elif self.unsupervised_loss == 'semco':
                    cat.extend([self.transforms(un_image)])
                    if len(self.rotation_labels) > 1:
                        un_image_90 = self.transforms(self.rotate_img(un_image, 90))
                        un_image_180 = self.transforms(self.rotate_img(un_image, 180))
                        un_image_270 = self.transforms(self.rotate_img(un_image, 270))
                        cat.extend([un_image_90, un_image_180, un_image_270])
                    un_r_labels.extend(self.rotation_labels)
                    cat.extend([self.strong_transforms(un_image)])
                    un_r_labels.extend([0])
                else:
                    if np.random.randint(2):
                        un_image = np.fliplr(un_image).copy()
                    cat.extend([self.transforms(un_image)])
                    if len(self.rotation_labels) > 1:
                        un_image_90 = self.transforms(self.rotate_img(un_image, 90))
                        un_image_180 = self.transforms(self.rotate_img(un_image, 180))
                        un_image_270 = self.transforms(self.rotate_img(un_image, 270))
                        cat.extend([un_image_90, un_image_180, un_image_270])
                    un_r_labels.extend(self.rotation_labels)

            un_images = torch.stack(cat) * 2 - 1
        else:
            un_images = torch.zeros(1)
            un_r_labels = self.rotation_labels

        return images, torch.ones(len(r_labels), dtype=torch.long)*int(self.labels[item]), \
               torch.LongTensor(r_labels), torch.Tensor(embs), un_images, torch.LongTensor(un_r_labels)

    def __len__(self):
        return self.size


class SemiRotationMiniImagenetPkl(Dataset):
    tasks_type = "clss"
    name = "miniimagenet"
    split_paths = {"train": "train", "val": "val", "valid": "val", "test": "test"}
    episodic = False
    c = 3
    h = 84
    w = 84

    def __init__(self, data_root, split, transforms, transforms_strong=None, data_group='train_label', rotation_labels=[0, 1, 2, 3],
                 unlabeled_size=0, unsupervised_loss='rotation', n_distract=0, **kwargs):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        im_size = 84
        self.unsupervised_loss = unsupervised_loss
        self.unlabeled_size = unlabeled_size
        # self.un_transforms = get_simsiam_trans(im_size)
        self.data_root = os.path.join(data_root, "mini-imagenet-cache-%s.pkl")
        t_split = split.split('-')
        if len(t_split) > 1:
            # flag to split the dataset to supervised and unsupervised
            split_semi = split  # such as 'train-K-002'
            split = t_split[0]

        with open(self.data_root % self.split_paths[split], 'rb') as infile:
            data = pkl.load(infile)
        self.features = data["image_data"]
        label_names = data["class_dict"].keys()
        self.labels = np.zeros((self.features.shape[0],), dtype=int)
        for i, name in enumerate(sorted(label_names)):
            self.labels[np.array(data['class_dict'][name])] = i
        if len(t_split) > 1:
            self.features, self.labels = load_semi_data_v2(self.data_root, self.features, self.labels,
                           split_semi, data_group)

        # add distract samples
        if n_distract > 0 and data_group == 'train_unlabel':
            with open(os.path.join(data_root, 'tiered_imagenet_distract_nc{}.pkl'.format(n_distract)),
                      'rb') as infile:
                data = pkl.load(infile)
            select_features = data['features']
            select_labels = data['targets'] + len(label_names)
            self.features = np.concatenate((self.features, select_features))
            self.labels = np.concatenate((self.labels, select_labels))

        del (data)
        self.transforms = transforms
        self.size = len(self.features)
        self.rotation_labels = rotation_labels
        self.labels_weight = np.ones_like(self.labels)
        # # load word embedding for each class name
        # self.embs = load_embs(data_root, label_names, self.split_paths[split])
        # self.embs = np.array(self.embs)
        # # fix match semi-supervise
        # self.embs_dict, self.class_names = load_embs_dict(data_root, label_names, self.split_paths[split])
        # if unsupervised_loss == 'semco':
        #     # self.transforms = get_weak_trans(im_size)
        #     self.strong_transforms = transforms_strong

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

        # if self.unsupervised_loss == 'semco':
        #     image_strong = self.strong_transforms(image)
        #     cat.extend([image_strong])
        #     # emb_list.extend([emb])
        #     r_labels.extend([0])

        images = torch.stack(cat) * 2 - 1
        # embs = torch.stack(emb_list)

        return images, torch.ones(len(r_labels), dtype=torch.long) * int(self.labels[item]), \
               torch.LongTensor(r_labels), torch.Tensor(image.copy())
        #, torch.Tensor(embs)

    def __len__(self):
        return self.size
