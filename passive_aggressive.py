#!/usr/bin/env python
#-*-coding:utf-8-*-

import numpy as np

class PassiveAggressive:
    def __init__(self, m):
        self.t = 0
        self.w = np.ones(m)

    def update(self, y, x):
        l = max([0, 1-y*np.dot(self.w, x)])
        eta = l/np.dot(x, x)
        self.w += eta*y*x
        return 1 if l == 0 else 0
