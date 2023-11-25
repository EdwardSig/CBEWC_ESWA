import random

import numpy as np


def select_negative_items(batch_history_data, nb):
    data = np.array(batch_history_data)
    idx = np.zeros_like(data)
    for i in range(data.shape[0]):
        items = np.where(data[i] != 0)[0].tolist()
        idx[i][items] = 1
    for i in range(data.shape[0]):
        items = np.where(data[i] == 0)[0].tolist()
        tmp_zr = random.sample(items, nb)
        idx[i][tmp_zr] = 1
    return idx