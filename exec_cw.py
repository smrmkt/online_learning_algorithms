#!/usr/bin/env python
#-*-coding:utf-8-*-

from evaluator import Evaluator
from loader import Loader
import matplotlib.pyplot as plt
from confidence_weighted import ConfidenceWeighted


def graph_plot(plt_obj, show=False):
    plt_obj.ylim(0, 1)
    plt_obj.xlabel("Number of trials")
    plt_obj.ylabel("Accuracy")
    plt_obj.legend(["CW", "CW1", "CW2"], loc="lower right")
    if show is True:
        plt_obj.show()
    else:
        plt_obj.figure()


if __name__ == '__main__':
    # construct passive-aggressive model
    cw = list()
    cw.append(ConfidenceWeighted(123))
    cw.append(ConfidenceWeighted(123, 0.30))
    cw.append(ConfidenceWeighted(123, 0.50))

    # training phase
    loader = Loader('a1a', 123, 30956, 1605)
    y_vec, feats_vec = loader.load_train()
    for i in range(len(cw)):
        evaluator = Evaluator(cw[i], y_vec, feats_vec)
        evaluator.update()
        plt.plot(evaluator.accuracy)
    graph_plot(plt)

    # test phase
    y_vec, feats_vec = loader.load_test()
    for i in range(len(cw)):
        evaluator = Evaluator(cw[i], y_vec, feats_vec)
        evaluator.predict()
        plt.plot(evaluator.accuracy)
    graph_plot(plt, show=True)
