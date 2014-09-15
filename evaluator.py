#!/usr/bin/env python
#-*-coding:utf-8-*-

import enum
import numpy as np


class Evaluator:
    CalcType = enum.Enum("CalcType", ["update", "predict"])

    def __init__(self, model, y_vec, feats_vec):
        self.model = model
        self.count = len(y_vec)
        self.y_vec = y_vec
        self.feats_vec = feats_vec
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.accuracy = []
        self.precision = []
        self.recall = []

    def _calc(self, calc_type):
        tp, fp, fn, tn = 0, 0, 0, 0
        accuracy, precision, recall = [], [], []
        for i in range(self.count):
            if calc_type == Evaluator.CalcType.update:
                ret = self.model.update(self.y_vec[i], self.feats_vec[i])
            elif calc_type == Evaluator.CalcType.predict:
                ret = 1 if self.y_vec[i]*self.model.predict(self.feats_vec[i]) > 0 else 0
            tp += 1 if ret == 1 and self.y_vec[i] == 1 else 0
            fp += 1 if ret == 0 and self.y_vec[i] == -1 else 0
            fn += 1 if ret == 0 and self.y_vec[i] == 1 else 0
            tn += 1 if ret == 1 and self.y_vec[i] == -1 else 0
            print tp, fp, fn, tn, float(tp+tn)/(tp+fp+fn+tn)
            accuracy.append(float(tp+tn)/(tp+fp+fn+tn))
            precision.append(float(tp)/(tp+fp) if tp+fp > 0 else 0.0)
            recall.append(float(tp)/(tp+fn) if tp+fn > 0 else 0.0)
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall

    def update(self):
        self._calc(Evaluator.CalcType.update)

    def predict(self):
        self._calc(Evaluator.CalcType.predict)
