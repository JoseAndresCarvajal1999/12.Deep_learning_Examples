import numpy as np


def sigmoid():
    return (__eval_sigmoid, __eval_sigmoid_diff)


def __eval_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def __eval_sigmoid_diff(x):
    return __eval_sigmoid(x) * (1.0 - __eval_sigmoid(x))


def tanh():
    return (__eval_tanh, __eval_tanh_diff)


def __eval_tanh(x):
    return np.tanh(x)


def __eval_tanh_diff(x):
    return 1 - __eval_tanh(x) ** 2


def linear():
    return (__eval_linear, __eval_linear_diff)


def __eval_linear(x):
    return x


def __eval_linear_diff(x):
    return 1.0


def relu():
    return (__eval_relu, __eval_relu_diff)


def __eval_relu(x):
    return np.maximum(x, 0)


def __eval_relu_diff(x):
    return (0 if x < 0 else 1)
