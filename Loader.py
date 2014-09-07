#!/usr/bin/env python
#-*-coding:utf-8-*-

import numpy as np


class Loader:
    def __init__(self, data, feats_dim, n_train, n_test):
        self.data = data
        self.feats_dim = feats_dim
        self.n_train = n_train
        self.n_test = n_test

    def load_train(self):
        return self._load('data/%s.train.txt' % self.data, self.n_train)

    def load_test(self):
        return self._load('data/%s.test.txt' % self.data, self.n_test)

    def _load(self, path, n):
        y_vec = []
        feats_vec = np.zeros((n, self.feats_dim))
        row = 0
        for l in open(path):
            items = l.split(' ')
            y_vec.append(int(items[0]))
            for i in range(1, len(items)-1):
                kv = items[i].split(":")
                k = int(kv[0])-1
                v = int(kv[1])
                feats_vec[row, k] = v
            row += 1
        return y_vec, feats_vec
