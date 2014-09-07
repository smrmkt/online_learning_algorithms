#!/usr/bin/env python
#-*-coding:utf-8-*-

from loader import Loader
from passive_aggressive import PassiveAggressive


if __name__ == '__main__':
    # train data
    loader = Loader('a1a', 123, 30956, 1605)
    y_vec, feats_vec = loader.load_train()
    pa = PassiveAggressive(feats_vec.shape[1])
    pa.train(y_vec, feats_vec)

    # test data
    y_vec, feats_vec = loader.load_test()
    count = 0
    for i in range(len(y_vec)):
        count += 1 if y_vec[i]*pa.predict(feats_vec[i]) > 0 else 0
    print count, len(y_vec), "{0:.2f}%".format(100.0*count/len(y_vec))







