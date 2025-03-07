import random

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def read2dict(filepath1, filepath2):
    train_set_dict, test_set_dict = {}, {}
    df = pd.read_csv(filepath1).iloc[:, :3] - 1

    df = df.values.tolist()
    df2 = pd.read_csv(filepath2).iloc[:, :3] - 1
    df2 = df2.values.tolist()

    train_set, test_set = df, df2
    for uid, iid, score in train_set:
        uid = int(uid)
        iid = int(iid)
        train_set_dict.setdefault(uid, {}).setdefault(iid, round(score + 1, 5))
    for uid, iid, score in test_set:
        uid = int(uid)
        iid = int(iid)
        test_set_dict.setdefault(uid, {}).setdefault(iid, round(score + 1, 5))

    return train_set_dict, test_set_dict


def dict2matrix(train_set_dict, test_set_dict, nb_user, nb_item):
    train_set, test_set = np.zeros(shape=(nb_user, nb_item)), np.zeros(shape=(nb_user, nb_item))
    for u in train_set_dict.keys():
        for i in train_set_dict[u].keys():
            train_set[u][i] = train_set_dict[u][i]
    for u in test_set_dict.keys():
        for i in test_set_dict[u]:
            test_set[u][i] = test_set_dict[u][i]

    return train_set, test_set


class M_Dataset(Dataset):
    def __init__(self, train_set):
        self.train_set = train_set

    def __getitem__(self, idx):
        purchase_vec = torch.tensor(self.train_set[idx], dtype=torch.float)
        uid = torch.tensor([idx, ], dtype=torch.long)
        return purchase_vec, uid

    def __len__(self):
        return len(self.train_set)
