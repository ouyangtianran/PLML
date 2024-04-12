import pickle as pkl
from pathlib import Path
from utils.semco.labels2wv import get_labels2wv_dict
# model = pickle.load(Path('../data/numberbatch-en-19.08_128D.dict.pkl').open('rb'))
# a=0
label_path = '../data/tiered-imagenet/train_labels.pkl'
with open(label_path, 'rb') as infile:
    labels = pkl.load(infile, encoding="bytes")['label_specific_str']

# labels = ['bicycle', 'apple', 'banana', 'motorbike', 'plane']
wv_dict_path = '../data/numberbatch-en-19.08_128D.dict.pkl'
class_embs = get_labels2wv_dict(labels, wv_dict_path)
a=0