import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader

from basic_evaluate import model_test
from basic_until import read2dict, dict2matrix, setup_seed, M_Dataset
from model import CDAE
from negative_items import select_negative_items


def train(nb_user, nb_item, nb_hidden, epoches,
          train_dataloader, w_dataloader,
          lr, nb_mask, train_set, test_set_dict,
          top_k):
    epoche_list, precision_list = [], []
    model = CDAE(nb_item, nb_user, nb_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    p_lst = []
    r_lst = []
    ndcg_lst = []
    for e in range(epoches):
        model.train()
        for index, data in enumerate(zip(train_dataloader, w_dataloader)):
            data = list(data)
            purchase_vec = list(data)[0][0]
            uid = list(data)[0][1]
            w_purchase_vec = list(data)[1][0]
            mask_vec = torch.tensor(select_negative_items(purchase_vec, nb_mask))
            out = model(uid, purchase_vec) * mask_vec  #
            loss = torch.sum(((out - purchase_vec).square()) * w_purchase_vec)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if (e + 1) % 5 == 0:
            print(e + 1, '\t', '==' * 24)
            precision, recall, NDCG = model_test(model, test_set_dict, train_set, top_k=top_k)
            p_lst.append(precision)
            r_lst.append(recall)
            ndcg_lst.append(NDCG)
            epoche_list.append(e + 1)
            precision_list.append(precision)


if __name__ == '__main__':
    nb_hidden = 256
    a = pd.read_csv('dataset/WD150/new_train_set.csv')
    b = a[['userId', 'movieId', 'rating', 'timestamp']]
    a = b
    setup_seed(4)

    nb_user = a["userId"].nunique()
    nb_item = a["movieId"].nunique()

    implicit_train_set_file = 'dataset/WD150/p_matrix.csv'
    implicit_mean_test_set_file = 'dataset/WD150/test.csv'
    prefer_weight_file = 'dataset/WD150/prefer_matrix.pkl'

    train_set_dict, test_set_dict = read2dict(implicit_train_set_file,
                                              implicit_mean_test_set_file)
    train_set, test_set = dict2matrix(train_set_dict, test_set_dict, nb_user=nb_user, nb_item=nb_item)

    with open(prefer_weight_file, 'rb') as f:
        w_set2 = pickle.load(f)

    dataset = M_Dataset(train_set)
    w_set = M_Dataset(w_set2)
    train_dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=False)
    w_dataloader = DataLoader(dataset=w_set, batch_size=256, shuffle=False)
    print('start train')
    train(nb_user, nb_item, nb_hidden, epoches=80, train_dataloader=train_dataloader, w_dataloader=w_dataloader,
          lr=7e-3, nb_mask=5000, train_set=train_set, test_set_dict=test_set_dict, top_k=5)
