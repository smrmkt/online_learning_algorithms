#!/usr/bin/env python
#-*-coding:utf-8-*-

from loader import Loader
from passive_aggressive import PassiveAggressive
from passive_aggressive_1 import PassiveAggressive1
from passive_aggressive_2 import PassiveAggressive2


if __name__ == '__main__':
    # train data
    loader = Loader('a1a', 123, 30956, 1605)
    y_vec, feats_vec = loader.load_train()
    pa = PassiveAggressive(feats_vec.shape[1])
    pa1 = PassiveAggressive1(feats_vec.shape[1])
    pa2 = PassiveAggressive2(feats_vec.shape[1])
    pa.train(y_vec, feats_vec)
    pa1.train(y_vec, feats_vec)
    pa2.train(y_vec, feats_vec)

    # test data
    y_vec, feats_vec = loader.load_test()
    count = [0, 0, 0]
    for i in range(len(y_vec)):
        count[0] += 1 if y_vec[i]*pa.predict(feats_vec[i]) > 0 else 0
        count[1] += 1 if y_vec[i]*pa1.predict(feats_vec[i]) > 0 else 0
        count[2] += 1 if y_vec[i]*pa2.predict(feats_vec[i]) > 0 else 0
    print count[0], len(y_vec), "{0:.2f}%".format(100.0*count[0]/len(y_vec))
    print count[1], len(y_vec), "{0:.2f}%".format(100.0*count[1]/len(y_vec))
    print count[2], len(y_vec), "{0:.2f}%".format(100.0*count[2]/len(y_vec))







