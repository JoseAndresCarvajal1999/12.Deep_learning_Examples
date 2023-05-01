import numpy as np


def encode_labels(labels):
    classes = np.unique(labels)
    bin_classes = np.zeros(shape=(labels.size, classes.size))
    for i, label in enumerate(labels):
        bin_classes[i, label] = 1
    return bin_classes
