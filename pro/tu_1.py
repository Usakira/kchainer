# -*- coding:utf-8 -*-
# python3

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

if __name__ == "__main__":
    x_data = np.array([5], dtype=np.float32)
    print(x_data)
    x = Variable(x_data)

    y = x**2 - 2*x + 1
    print(y.data)
    print(y.backward())
