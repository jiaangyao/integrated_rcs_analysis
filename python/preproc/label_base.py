"""
This function contains base functionalities for preprocessing data labels. (Previously preproc/base)
"""
from itertools import product

import numpy as np
import polars as pl


def binarize_label(label):
    raise NotImplementedError("Not implemented yet")


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
        return np.array(label1), None
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


def create_labels_from_threshold(df, feature_column, threshold=0.5, out_column_name='Label', invert=False):
    """
    Create a label from a threshold value based on percentile.
    
    Args:
        df: polars DataFrame
        feature_column: column name to threshold
        threshold: fractional value between 0 and 1
        out_column_name: name for the new label column
        invert: if True, invert the label (i.e. below threshold -> 1, above threshold -> 0)
    """
    value = df.select(pl.col(feature_column).quantile(threshold, interpolation="midpoint")).item()
    if not invert:
        df = df.with_columns(pl.when(pl.col(feature_column) > value).then(1).otherwise(0).alias(out_column_name))
    else:
        df = df.with_columns(pl.when(pl.col(feature_column) < value).then(1).otherwise(0).alias(out_column_name))
    return df
