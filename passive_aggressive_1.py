#!/usr/bin/env python
#-*-coding:utf-8-*-

import numpy as np


class PassiveAggressive1:
    def __init__(self, feat_dim, c=0.1):
        self.t = 0
        self.w = np.ones(feat_dim)
        self.c = c

    def train(self, y_vec, feats_vec):
        for i in range(len(y_vec)):
            self.update(y_vec[i], feats_vec[i,])

    def predict(self, feats):
        return np.dot(self.w, feats)

    def update(self, y, feats):
        l = max([0, 1-y*np.dot(self.w, feats)])
        eta = min(self.c, l/np.dot(feats, feats))
        self.w += eta*y*feats
        self.t += 1
        return 1 if l == 0 else 0



