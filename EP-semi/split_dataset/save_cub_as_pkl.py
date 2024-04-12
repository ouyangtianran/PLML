import os
import h5py
import random
import numpy as np
import pickle as pkl
import json
from PIL import Image

def save_pkl(data_root, split):
    with open(os.path.join(data_root, "few_shot_lists", "%s.json" % split), 'r') as infile:
        metadata = json.load(infile)
    labels = np.array(metadata['image_labels'])
    labels_str = metadata['label_names']
    image_names = metadata['image_names']
    images = []
    for im_name in image_names:
        im = Image.open((im_name))
        im_re = im.resize((84, 84))
        image = np.array(im_re.convert("RGB"))
        images.append(image)
    new_data = {}
    new_data['labels'] = labels
    new_data['features'] = images
    new_data['label_names'] = labels_str

    save_name = 'cub-{}.pkl'.format(split)
    if os.path.exists(os.path.join(data_root, save_name)):
        print(os.path.join(data_root, save_name) + ' exist.')
    else:
        with open(os.path.join(data_root, save_name), 'wb') as handle:
            pkl.dump(new_data, handle, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    datapath = '../data'
    dataset = 'CUB_200_2011'
    datapath = os.path.join(datapath, dataset)
    split = 'base'
    nk = 1
    data_root = datapath
    for split in ['base', 'val', 'novel']:
        save_pkl(data_root, split)