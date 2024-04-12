import os
import h5py
import random
import numpy as np
import pickle as pkl


def split_dataset(data_root, split, seed, train_num_per_class, nks, base_path=None):
    with open(data_root % split, 'rb') as infile:
        data = pkl.load(infile)
    features = data["image_data"]
    label_names = data["class_dict"].keys()
    class_dict = data["class_dict"]
    if base_path:
        with open(base_path, 'rb') as infile:
            data_index = pkl.load(infile)
            # labeled features and labels
        ind_s = data_index['supervised_index']
        ind_u = data_index['unsupervised_index']
        ind_t = data_index['test_index']
        sp_base = base_path.split('-')
        npc_str = sp_base[-1].split('.')
        npc = int(npc_str[0])
        n_u = train_num_per_class - npc
        for i, name in enumerate(sorted(label_names)):
            p_n1 = np.array(class_dict[name])
            n_t = len(p_n1) - train_num_per_class
            p_n1[:npc] = ind_s[i*npc:(i+1)*npc]
            p_n1[npc:train_num_per_class] = ind_u[i*n_u: (i+1)*n_u]
            p_n1[train_num_per_class:] = ind_t[i*n_t : (i+1)*n_t]
            class_dict[name] = p_n1
    else:
        # random permutation class
        for i, name in enumerate(sorted(label_names)):
            n1 = np.array(class_dict[name])
            # print('shape {}'.format(n1.shape))
            p_n1 = np.random.permutation(n1)
            class_dict[name] = p_n1
    # labels = np.zeros((features.shape[0],), dtype=int)
    for nk in nks:#[1, 5, 10]:
        supervised_index = []
        unsupervised_index = []
        test_index = []

        for i, name in enumerate(sorted(label_names)):
            # labels[np.array(data['class_dict'][name])] = i
            # p_n1 = np.array(class_dict[name])
            p_n1 = class_dict[name]
            # print('shape {}'.format(n1.shape))
            # p_n1 = np.random.permutation(n1)
            s_ind = p_n1[:nk]
            u_ind = p_n1[nk:train_num_per_class]
            t_ind = p_n1[train_num_per_class:]
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

        save_name = '{}-test-seed-{}-K-{:03d}'.format(split, seed, nk)
        if os.path.exists(data_root % save_name):
            print(data_root % save_name + ' exist.')
            continue
        with open(data_root % save_name, 'wb') as handle:
            pkl.dump(new_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
        print('Finish nk = {}'.format(nk))

def test_split(file_path):
    with open(file_path, 'rb') as infile:
        data = pkl.load(infile)
    print('supervised index: {} num: {}'.format(data['supervised_index'],len(data['supervised_index'])))
    print('unsupervised_index: {} num: {}'.format(data['unsupervised_index'],len(data['unsupervised_index'])))
    print('test_index: {} num {}'.format(data['test_index'], len(data['test_index'])))

datapath = './data'
dataset = 'mini-imagenet'
seed = 10
np.random.seed(seed)
num_per_class = 600
train_num_per_class = 500
test_num_per_class = num_per_class - train_num_per_class
datapath = os.path.join(datapath, dataset)
split = 'train'
nk=1
data_root = os.path.join(datapath, "mini-imagenet-cache-%s.pkl")

nks = [2, 6, 10, 20, int(0.1*train_num_per_class), int(0.2*train_num_per_class), int(0.5*train_num_per_class)]
# nks = [6, 10]
base_path = "data/mini-imagenet/backup/mini-imagenet-cache-train-test-seed-10-K-020.pkl"
split_dataset(data_root, split, seed, train_num_per_class, nks, base_path)

# file_path = './data/mini-imagenet/mini-imagenet-cache-test-seed-10-train-K-002.pkl'
# test_split(file_path)