import numpy as np
import os
import pickle as pkl


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

def load_semi_data_validation(data_root, features0, labels0, split, data_group):
    with open(data_root % split, 'rb') as infile:
        data_index = pkl.load(infile)
    # last 10 % labeled data as validation
    if data_group == 'train_label':
        # labeled features and labels
        ind = data_index['supervised_index']
        features = features0[ind]
        labels = labels0[ind]
    elif data_group == 'train_unlabel':
        # unlabeled features and labels
        ind = data_index['unsupervised_index']
        features = features0[ind]
        labels = labels0[ind]
    elif data_group == 'train_test' or data_group == 'train_validation':
        # test features and labels
        ind = data_index['test_index']
        features = features0[ind]
        labels = labels0[ind]
    else:
        raise NotImplementedError

    return features, labels


