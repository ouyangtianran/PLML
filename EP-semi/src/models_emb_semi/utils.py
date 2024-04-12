import torch
from src.utils import pairwise_distances
import torch.nn.functional as F
# from collections import OrderdDict

def remove_prefix(pretrain_static_dict, prefix='module.'):
    # remove the prefix 'module.'
    dict = {}
    for k, v in pretrain_static_dict.items():
        if prefix in k:
            k = k.strip(prefix)
        dict[k] = v
    return dict

def get_proto(pretrained_embs, embeddings, y):
    # combine visual and word to get proto
    u_y = torch.unique(y)
    # adjust word embedding space
    # emb = self.model.adjust_word(pretrained_embs)
    emb = pretrained_embs

    emb_new = [e for e in emb]
    for i in u_y:
        proto = embeddings[y == i]
        emb_new[i] = emb[i] * 0.5 + proto.mean(dim=0) * 0.5
    emb_new = torch.stack(emb_new)

    # # use word embedding as proto
    # emb_new = pretrained_embs
    return emb_new


def get_pseudo_label(embeddings_un, proto, remove_noise=None, y=None, y_u_gt=None):
    # remove_noise = self.exp_dict['remove_noise']
    # # dot similarity
    # logits = torch.einsum('nc,ck->nk', [embeddings_un, proto.T])
    # # apply temperature
    # logits /= self.T
    ######## l2 distance ##########
    dist = pairwise_distances(embeddings_un, proto, 'l2')
    logits = - dist
    with torch.no_grad():
        max_value, ind = torch.max(logits, dim=1)
    if remove_noise == 'half':
        _, ind_sort = torch.sort(max_value)
        num = len(ind_sort) // 2
        ind_select = ind_sort[:num]
        return logits[ind_select], ind[ind_select], embeddings_un[ind_select]
    elif remove_noise == 'keep_visual':
        # _, ind_sort = torch.sort(max_value)
        # num = len(ind_sort) // 2
        # ind_select = ind_sort[:num]
        # logits, ind, embeddings_un = logits[ind_select], ind[ind_select], embeddings_un[ind_select]
        u_y = torch.unique(y)
        ind_select = []
        for i in u_y:
            tt = torch.nonzero(ind == i)
            if len(tt) > 0:
                ind_select.append(tt)
        if len(ind_select) > 0:
            ind_select = torch.cat(ind_select).view(-1)
            if len(ind_select) < embeddings_un.shape[0]:
                print('len of ind_select {}'.format(len(ind_select)))
            return logits[ind_select], ind[ind_select], embeddings_un[ind_select]
        else:
            # return logits[:2], ind[:2], embeddings_un[:2]
            return [], [], []
    else:
        return logits, ind, embeddings_un


def top_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)  # return value, indices
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# unsupervised loss
# "unsupervised_loss": "semco",#["simsiam","word_proto","rotation","classifier_loss"]

def weight_class_loss(classifier, embeddings, r, r_w):
    rot = classifier(embeddings)
    loss = F.cross_entropy(rot, r) * r_w
    return loss


def weight_E_distance(visual_emb, word_emb, wew):
    # wew: "word_emb_weight"
    distance = visual_emb - word_emb
    distance = (distance ** 2).mean(dim=1)
    loss = distance.mean() * wew
    return loss
