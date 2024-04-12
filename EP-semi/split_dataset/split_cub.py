import os
import h5py
import random
import numpy as np
import pickle as pkl
import json

def split_dataset(data_root, split, seed, test_num_per_class, nks, base_path=None):

    with open(os.path.join(data_root, "few_shot_lists", "%s.json" % split), 'r') as infile:
        metadata = json.load(infile)
    labels = np.array(metadata['image_labels'])
    labels_str = metadata['label_names']
    n_label = len(labels_str)
    # labels = np.random.permutation(labels)

    indices = np.arange(len(labels))
    for nk in nks:#[1, 5, 10]:
        supervised_index = []
        unsupervised_index = []
        test_index = []

        for i in range(n_label):
            # labels[np.array(data['class_dict'][name])] = i
            # p_n1 = np.array(class_dict[name])
            p_n1 = indices[labels==i]
            # print('class: {}, shape: {}'.format(labels_str[i], p_n1.shape))
            # p_n1 = np.random.permutation(n1)
            s_ind = p_n1[:nk]
            u_ind = p_n1[nk:-test_num_per_class]
            t_ind = p_n1[-test_num_per_class:]
            supervised_index.append(s_ind)
            unsupervised_index.append(u_ind)
            test_index.append(t_ind)
        supervised_index = np.concatenate(supervised_index)
        unsupervised_index = np.concatenate(unsupervised_index)
        test_index = np.concatenate(test_index)
        # save indexs to pkl file
        new_data = {}
        new_data['supervised_index'] = supervised_index
        new_data['unsupervised_index'] = unsupervised_index
        new_data['test_index'] = test_index

        save_name = '{}-test-seed-{}-K-{:03d}.pkl'.format(split, seed, nk)
        if os.path.exists(os.path.join(data_root , save_name)):
            print(os.path.join(data_root , save_name) + ' exist.')
            continue
        with open(os.path.join(data_root , save_name), 'wb') as handle:
            pkl.dump(new_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
        print('Finish nk = {}'.format(nk))

def test_split(file_path):
    with open(file_path, 'rb') as infile:
        data = pkl.load(infile)
    print('supervised index: {} num: {}'.format(data['supervised_index'],len(data['supervised_index'])))
    print('unsupervised_index: {} num: {}'.format(data['unsupervised_index'],len(data['unsupervised_index'])))
    print('test_index: {} num {}'.format(data['test_index'], len(data['test_index'])))

datapath = '../data'
dataset = 'CUB_200_2011'
seed = 10
np.random.seed(seed)
num_per_class = 60
train_num_per_class = 50
test_num_per_class = num_per_class - train_num_per_class
datapath = os.path.join(datapath, dataset)
split = 'base'
nk=1
data_root = datapath

nks = [2, 6, 10, 20]#, int(0.5*train_num_per_class)
# nks = [6, 10]
# base_path = "data/mini-imagenet/backup/mini-imagenet-cache-train-test-seed-10-K-020.pkl"
base_path = None
split_dataset(data_root, split, seed, test_num_per_class, nks, base_path)

# file_path = './data/mini-imagenet/mini-imagenet-cache-test-seed-10-train-K-002.pkl'
# test_split(file_path)