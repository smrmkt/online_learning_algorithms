#!/usr/bin/env python
#-*-coding:utf-8-*-

import numpy as np
from passive_aggressive import PassiveAggressive

# load test data
def load_a1a():
    y = []
    x = np.zeros((30956, 123))
    row = 0
    for l in open('data/a1a.test.txt'):
        items = l.split(' ')
        y.append(int(items[0]))
        for i in range(1, len(items)-1):
            kv = items[i].split(":")
            k = int(kv[0])-1
            v = int(kv[1])
            x[row, k] = v
        row += 1
    return y, x


if __name__ == '__main__':
    y, x = load_a1a()
    correct = 0
    pa = PassiveAggressive(x.shape[1])
    for i in range(len(y)):
        correct += pa.update(y[i], x[i])
        if i % 1000 == 0:
            print i, correct
    print correct







