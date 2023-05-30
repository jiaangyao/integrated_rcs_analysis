from itertools import product

import numpy as np


def combine_labels(
    label1,
    label2,
    hashmap=None,
):
    """
    Combine two labels into one label, starting with 0
    :param label1: first label
    :param label2: second label
    :param n_class: number of classes in the data (for checking)
    :return: combined label
    """

    # first check if the labels are the same
    if np.all(label1 == label2):
        return np.array(label1)
    else:
        # get the cartesian product and figure out the number of unique values
        catesian_product = list(zip(label1, label2))

        # create the new label
        if hashmap is None:
            hashmap = create_hashmap(label1, label2)
        new_label = np.array([hashmap[x] for x in catesian_product])

        return new_label, hashmap


def create_hashmap(
    label1,
    label2,
):
    # create the hashmap
    hashmap = dict()

    # get the unique values
    for i in range(len(np.unique(label1))):
        for j in range(len(np.unique(label2))):
            hashmap[(i, j)] = i * len(np.unique(label2)) + j

    return hashmap
