#!/usr/bin/env python
#-*-coding:utf-8-*-

from loader import Loader
import matplotlib.pyplot as plt
from passive_aggressive import PassiveAggressive
from passive_aggressive_1 import PassiveAggressive1
from passive_aggressive_2 import PassiveAggressive2


if __name__ == '__main__':
    # construct passive-aggressive model
    pa = list()
    pa.append(PassiveAggressive(123))
    pa.append(PassiveAggressive1(123))
    pa.append(PassiveAggressive2(123, 0.01))

    # training phase
    loader = Loader('a1a', 123, 30956, 1605)
    y_vec, feats_vec = loader.load_train()
    count = [0, 0, 0]
    accuracy = [[], [], []]
    for i in range(len(y_vec)):
        for j in range(3):
            count[j] += pa[j].update(y_vec[i], feats_vec[i])
            accuracy[j].append(100.0*count[j]/(i+1))
    for i in range(3):
        print count[i], len(y_vec), "{0:.2f}%".format(100.0*count[i]/len(y_vec))
        plt.plot(accuracy[i])
    plt.ylim(0, 100)
    plt.xlabel("Number of trials")
    plt.ylabel("Accuracy")
    plt.legend(["PA", "PA1", "PA2"])
    plt.show()

    # test phase
    y_vec, feats_vec = loader.load_test()
    count = [0, 0, 0]
    accuracy = [[], [], []]
    for i in range(len(y_vec)):
        for j in range(3):
            count[j] += 1 if y_vec[i]*pa[j].predict(feats_vec[i]) > 0 else 0
            accuracy[j].append(100.0*count[j]/(i+1))
    for i in range(3):
        print count[i], len(y_vec), "{0:.2f}%".format(100.0*count[i]/len(y_vec))
        plt.plot(accuracy[i])
    plt.ylim(0, 100)
    plt.xlabel("Number of trials")
    plt.ylabel("Accuracy")
    plt.legend(["PA", "PA1", "PA2"])
    plt.show()

