import os
import h5py
import random
import numpy as np
import pickle as pkl
datapath = './data'
dataset = 'mini-imagenet'
num_per_class = 600
datapath = os.path.join(datapath, dataset)
split = 'train'
nk=1
data_root = os.path.join(datapath, "mini-imagenet-cache-%s.pkl")

# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


nks = [2, 20, int(0.1*num_per_class), int(0.2*num_per_class), int(0.5*num_per_class), int(0.9*num_per_class)]

with open(data_root % split, 'rb') as infile:
    data = pkl.load(infile)
features = data["image_data"]
label_names = data["class_dict"].keys()
# labels = np.zeros((features.shape[0],), dtype=int)
for nk in nks:#[1, 5, 10]:
    supervised_index = []
    unsupervised_index = []

    for i, name in enumerate(sorted(label_names)):
        # labels[np.array(data['class_dict'][name])] = i
        n1 = np.array(data['class_dict'][name])
        # print('shape {}'.format(n1.shape))
        p_n1 = np.random.permutation(n1)
        s_ind = p_n1[:nk]
        u_ind = p_n1[nk:]
        supervised_index.append(s_ind)
        unsupervised_index.append(u_ind)
    supervised_index = np.concatenate(supervised_index)
    unsupervised_index = np.concatenate(unsupervised_index)
    # save indexs to pkl file
    new_data = {}
    new_data['supervised_index'] = supervised_index
    new_data['unsupervised_index'] = unsupervised_index

    save_name = '{}-K-{:03d}'.format(split, nk)
    with open(data_root % save_name, 'wb') as handle:
        pkl.dump(new_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print('Finish nk = {}'.format(nk))


    # with h5py.File(os.path.join(datapath, split + '_data.hdf5'), 'r') as f:
    #     with h5py.File(os.path.join(datapath, split + '_n_{:02d}_data.hdf5'.format(nk)), 'w') as f2:
    #         datasets = f['datasets']
    #         g1 = f2.create_group('datasets')
    #         for k in datasets.keys():
    #             n1 = np.array(datasets[k][()])
    #             # print('shape {}'.format(n1.shape))
    #             n = len(n1)
    #             ind = random.sample(range(n), nk)
    #             c = n1[ind]
    #             # if nk==1:
    #             #     c = np.expand_dims(c, axis=0)
    #             # print('{} shape {}'.format(k, c.shape))
    #             g1.create_dataset(k, data=c)
    #             print('classes {}'.format(k))
    #
    #
    # with h5py.File(os.path.join(datapath, split + '_n_{:02d}_data.hdf5'.format(nk)), 'r') as f:
    #     datasets = f['datasets']
    #     classes = [datasets[k][()] for k in datasets.keys()]