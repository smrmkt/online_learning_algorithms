#!/usr/bin/env python
#-*-coding:utf-8-*-

import numpy as np


class PassiveAggressive:
    def __init__(self, feat_dim):
        self.t = 0
        self.w = np.ones(feat_dim)

    @staticmethod
    def _get_eta(l, feats):
        return l/np.dot(feats, feats)

    def train(self, y_vec, feats_vec):
        for i in range(len(y_vec)):
            self.update(y_vec[i], feats_vec[i,])

    def predict(self, feats):
        return np.dot(self.w, feats)

    def update(self, y, feats):
        l = max([0, 1-y*np.dot(self.w, feats)])
        eta = self._get_eta(l, feats)
        self.w += eta*y*feats
        self.t += 1
        return 1 if l == 0 else 0



