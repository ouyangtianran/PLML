import numpy as np
import os


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