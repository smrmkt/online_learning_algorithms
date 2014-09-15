#!/usr/bin/env python
#-*-coding:utf-8-*-

from loader import Loader
from passive_aggressive import PassiveAggressive
from passive_aggressive_1 import PassiveAggressive1


if __name__ == '__main__':
    # train data
    loader = Loader('a1a', 123, 30956, 1605)
    y_vec, feats_vec = loader.load_train()
    pa = PassiveAggressive(feats_vec.shape[1])
    pa.train(y_vec, feats_vec)
    pa1 = PassiveAggressive1(feats_vec.shape[1])
    pa1.train(y_vec, feats_vec)

    # test data
    y_vec, feats_vec = loader.load_test()
    count_pa = 0
    count_pa1 = 0
    for i in range(len(y_vec)):
        count_pa += 1 if y_vec[i]*pa.predict(feats_vec[i]) > 0 else 0
        count_pa1 += 1 if y_vec[i]*pa1.predict(feats_vec[i]) > 0 else 0
    print count_pa, len(y_vec), "{0:.2f}%".format(100.0*count_pa/len(y_vec))
    print count_pa1, len(y_vec), "{0:.2f}%".format(100.0*count_pa1/len(y_vec))







