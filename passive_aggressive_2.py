#!/usr/bin/env python
#-*-coding:utf-8-*-

import numpy as np
from passive_aggressive import PassiveAggressive


class PassiveAggressive2(PassiveAggressive):
    def __init__(self, feat_dim, c=0.1):
        self.c = c
        PassiveAggressive.__init__(self, feat_dim)

    def _get_eta(self, l, feats):
        return l/(np.dot(feats, feats)+1/(2*self.c))
