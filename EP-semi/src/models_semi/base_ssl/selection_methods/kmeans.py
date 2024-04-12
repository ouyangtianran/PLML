import numpy as np
import torch
from sklearn.cluster import KMeans


def kmeans_select(episode_dict, support_size_max=None):
    S = episode_dict["support_so_far"]["samples"].copy()
    Q = episode_dict["query"]["samples"].copy()
    U = episode_dict["unlabeled"]["samples"].copy()
    A = np.concatenate([S, U], axis=0)   # (_, d)

    S_labels = episode_dict["support_so_far"]["labels"].copy()
    nb_unlabeled, dim = U.shape
    n_classes = int(S_labels.max() + 1)

    X_centers = np.zeros((n_classes, dim))
    for cls in range(n_classes):
        X_centers[cls] = np.mean(S[S_labels == cls], axis=0)

    km = KMeans(n_clusters=n_classes, init=X_centers, random_state=1996)
    km.fit(A)
    A_logits = km.transform(A)   # distances w.r.t each class centers
    U_logits = A_logits[-nb_unlabeled:]
    episode_dict['unlabeled']['labels'] = U_logits.argmin(axis=1)

    if support_size_max is None:
        # choose all the unlabeled examples
        return np.arange(U.shape[0])
    else:
        # score each
        score_list = U_logits.max(axis=1)
        return score_list.argsort()[-support_size_max:]



def softmax(x):
    max = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x

def MultiFactorClustering(episode_dict, n_loops=10, lam = 0.01):
    S = episode_dict["support_so_far"]["samples"].copy()
    Q = episode_dict["query"]["samples"].copy()
    U = episode_dict["unlabeled"]["samples"].copy()

    nb_support, dim = S.shape
    nb_query, dim = Q.shape
    nb_unlabeled, dim = U.shape

    S_labels = episode_dict["support_so_far"]["labels"].copy()
    A = np.concatenate([S, U], axis=0)
    n_classes = int(S_labels.max() + 1)
    nb_sample = int(nb_support/n_classes)

    X_centers = np.zeros((n_classes, dim))
    for cls in range(n_classes):
        X_centers[cls] = np.mean(S[S_labels == cls], axis=0)

    pre = 0.
    for i in range(n_loops):
        X = []
        for cls in range(n_classes):
            X.append(S.T[:, S_labels == cls]) # ith class labeled data
            X.append( X_centers[cls].reshape(-1, 1))
        X = np.concatenate(X, axis=1)
        I = np.identity(int((nb_sample + 1) * n_classes))

        P = X.T.dot(X) + lam * I  # FT*F + lambda*I
        P = np.linalg.inv(P).dot(X.T)
        beta = P.dot(A.T).reshape(n_classes, -1, nb_unlabeled+nb_support)
        code_x = np.zeros((n_classes, dim, nb_unlabeled+nb_support))
        X = X.T.reshape(n_classes, -1, dim)
        for cls in range(n_classes):
            code_x[cls] = X[cls].T.dot(beta[cls])
        err = A.T - code_x

        dist = np.sum(err ** 2, axis=1)
        A_labels = dist.T.argmin(axis=1)

        for j in range(n_classes):
            X_centers[j] = np.mean(A[A_labels == j], axis=0)

        if pre == dist.mean():
            break
        else:
            pre = dist.mean()
        print('{}: {}'.format(i, dist.mean()))


    U_logits = dist.T[-nb_unlabeled:]
    U_logits = -np.log(U_logits)
    U_probs = softmax(U_logits / 0.1)
    U_labels =  dist.T.argmin(axis=1)

    episode_dict['unlabeled']['labels'] = U_labels[-nb_unlabeled:]

    return np.arange(U.shape[0])

def MultiFactorClustering_logit(S, Q, S_labels, n_classes, n_loops=10, lam = 0.01):
    nb_support, d = S.size()
    nb_query, d = Q.size()

    A = torch.cat([S, Q], dim=0)
    n_classes = int(S_labels.max() + 1)
    nb_sample = int(nb_support/n_classes)


    X_centers = torch.zeros((n_classes, d)).cuda()

    pre = 0.
    with torch.no_grad():
        for cls in range(n_classes):
            X_centers[cls] = torch.mean(S[S_labels == cls], dim=0)
        for i in range(n_loops):
            ## all class
            X = []
            for cls in range(n_classes):
                X.append(S.t()[:, S_labels == cls]) # ith class labeled data
                X.append( X_centers[cls].view(-1, 1))
            X = torch.cat(X, dim=1)
            I = torch.eye(int((nb_sample + 1) * n_classes)).cuda()

            P = X.t().mm(X) + lam * I  # FT*F + lambda*I
            P = torch.inverse(P).mm(X.t())
            beta = P.mm(A.t()).view(n_classes, -1, nb_query+nb_support) ## C*n
            code_x = torch.zeros((n_classes, d, nb_query+nb_support)).cuda()
            X = X.t().view(n_classes, -1, d)
            for cls in range(n_classes):
                code_x[cls] = X[cls].t().mm(beta[cls])
            err = A.t() - code_x
            ##

            dist = torch.sum(err ** 2, dim=1)  # C*n
            # A_logits =

            _, A_labels = dist.t().min(dim=1)

            for j in range(n_classes):
                X_centers[j] = torch.mean(A[A_labels == j], dim=0)

            if pre == dist.mean():
                break
            else:
                pre = dist.mean()
    for j in range(n_classes):
        X_centers[j] = torch.mean(A[A_labels == j], dim=0)
    X = []
    for cls in range(n_classes):
        X.append(S.t()[:, S_labels == cls])  # ith class labeled data
        X.append(X_centers[cls].view(-1, 1))
    X = torch.cat(X, dim=1)
    I = torch.eye(int((nb_sample + 1) * n_classes)).cuda()
    P = X.t().mm(X) + lam * I  # FT*F + lambda*I
    P = torch.inverse(P).mm(X.t())
    beta = P.mm(Q.t()).view(n_classes, -1, nb_query)
    code_x = torch.zeros((n_classes, d, nb_query)).cuda()
    X = X.t().view(n_classes, -1, d)
    for cls in range(n_classes):
        code_x[cls] = X[cls].t().mm(beta[cls])
    err = Q.t() - code_x
    ##

    dist = torch.sum(err ** 2, dim=1)


    Q_logits = dist.t()
    Q_logits = -torch.log(Q_logits)
    Q_logits = torch.softmax(Q_logits / 0.1, dim=1)
    _, Q_labels =  Q_logits.max(dim=1)

    return Q_logits, Q_labels

