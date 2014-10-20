#!/usr/bin/env python
#-*-coding:utf-8-*-

from math import sqrt
import numpy as np
from scipy.stats import norm

class ConfidenceWeighted():
    def __init__(self, feat_dim, eta=0.90):
        self.t = 0
        self.m = np.ones(feat_dim)
        self.s = np.diag([1.0]*feat_dim)
        self.eta = eta
        self.phi = norm.cdf(self.eta)**(-1)
        self.psi = 1.0+(self.phi**2)/2.0
        self.zeta = 1.0+self.phi**2

    def predict(self, feats):
        return np.dot(self.m, feats)

    def update(self, y, feats):
        # parameter calculation
        v = np.dot(np.dot(feats, self.s), feats)
        m = y*(np.dot(self.m, feats))
        part = sqrt((m**2)*(self.phi**4)/4.0+v*(self.phi**2)*self.zeta)
        alpha = max(0.0, 1.0/(v*self.zeta)*(-m*self.psi+part))
        u = 0.25*((-alpha*v*self.phi+sqrt((alpha**2)*(v**2)*(self.phi**2)+4.0*v))**2)
        beta = (alpha*self.phi)/(sqrt(u)+v*alpha*self.phi)
        # update parameters
        self.t += 1
        self.m += alpha*y*np.dot(self.s, feats)
        self.s -= beta*np.dot(np.matrix(np.dot(self.s, feats).T*feats), self.s)
        return 1 if np.dot(self.m, feats) > 0 else 0