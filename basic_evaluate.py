import numpy as np
import torch


def get_ndcg(l1, l2):
    hit = []
    dcg = 0
    idcg = 0
    for i in l1:
        if i in l2:
            hit.append(1)
        else:
            hit.append(0)
    if len(l2) >= len(l1):
        ihit = len(l1)
    else:
        ihit = len(l2)
    for i in range(len(hit)):
        dcg += np.divide(np.power(2, hit[i]) - 1, np.log2(i + 2))
    for i in range(ihit):
        idcg += np.divide(np.power(2, 1) - 1, np.log2(i + 2))
    ndcg = dcg / idcg
    return ndcg


def model_test(model, test_set_dict, train_set, top_k):
    model.eval()
    users = list(test_set_dict.keys())
    input_data = torch.tensor(train_set[users], dtype=torch.float)
    uids = torch.tensor(users, dtype=torch.long).view(-1, 1)
    out = model(uids, input_data)
    out = (out - 999 * input_data).detach().numpy()
    precisions = 0
    recalls = 0
    hits = 0
    total_purchase_nb = 0
    ndcg = []
    for i, u in enumerate(users):
        hit = 0
        tmp_list = [(idx, value) for idx, value in enumerate(out[i])]
        tmp_list = sorted(tmp_list, key=lambda x: x[1], reverse=True)[:top_k]
        a = []
        b = []
        for k, v in tmp_list:
            a.append(k)
        for i in test_set_dict[u]:
            b.append(i)
        ndcg.append(get_ndcg(a, b))
        for k, v in tmp_list:
            if k in test_set_dict[u]:
                hit += 1
        recalls += hit / len(test_set_dict[u])
        precisions += hit / top_k
        hits += hit
        total_purchase_nb += len(test_set_dict[u])
    recall = recalls / len(users)
    precision = precisions / len(users)
    NDCG = sum(ndcg) / len(ndcg)
    print('recall:{}, precision:{},NDCG:{}'.format(recall, precision, NDCG))
    return precision, recall, NDCG
