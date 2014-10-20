#!/usr/bin/env python
#-*-coding:utf-8-*-

from evaluator import Evaluator
from loader import Loader
import matplotlib.pyplot as plt
from passive_aggressive import PassiveAggressive
from passive_aggressive_1 import PassiveAggressive1
from passive_aggressive_2 import PassiveAggressive2


def graph_plot(plt_obj, show=False):
    plt_obj.ylim(0, 1)
    plt_obj.xlabel("Number of trials")
    plt_obj.ylabel("Accuracy")
    plt_obj.legend(["PA", "PA1", "PA2"], loc="lower right")
    if show is True:
        plt_obj.show()
    else:
        plt_obj.figure()


if __name__ == '__main__':
    # construct passive-aggressive model
    pa = list()
    pa.append(PassiveAggressive(123))
    pa.append(PassiveAggressive1(123))
    pa.append(PassiveAggressive2(123, 0.01))

    # training phase
    loader = Loader('a1a', 123, 30956, 1605)
    y_vec, feats_vec = loader.load_train()
    for i in range(len(pa)):
        evaluator = Evaluator(pa[i], y_vec, feats_vec)
        evaluator.update()
        plt.plot(evaluator.accuracy)
    graph_plot(plt)

    # test phase
    y_vec, feats_vec = loader.load_test()
    for i in range(len(pa)):
        evaluator = Evaluator(pa[i], y_vec, feats_vec)
        evaluator.predict()
        plt.plot(evaluator.accuracy)
    graph_plot(plt, show=True)
