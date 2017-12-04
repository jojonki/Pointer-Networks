# ref code
# https://github.com/guacomolia/ptr_net/blob/master/generate_data.py

import numpy as np
import random


def generate_single_seq(length=30, min_len=5, max_len=10):
    # https://medium.com/@devnag/pointer-networks-in-tensorflow-with-sample-code-14645063f264
    """ Generates a sequence of numbers of random length and inserts a sub-sequence oh greater numbers at random place
    Input:
    length: total sequence length
    min_len: minimum length of sequence
    max_len: maximum length of sequence
    Output:
    sequence of numbers, index of the start of greater numbers subsequence"""
    seq_before = [(random.randint(1, 5)) for x in range(random.randint(min_len, max_len))]
    seq_during = [(random.randint(6, 10)) for x in range(random.randint(min_len, max_len))]
    seq_after = [random.randint(1, 5) for x in range(random.randint(min_len, max_len))]
    seq = seq_before + seq_during + seq_after
    seq = seq + ([0] * (length - len(seq)))
    return (seq, len(seq_before), len(seq_before) + len(seq_during)-1)


def generate_set_seq(N):
    # generate Boundary tasks
    """Generates a set of N sequences of fixed length"""
    data = []
    starts = []
    ends = []
    for i in range(N):
        seq, ind_start, ind_end = generate_single_seq()
        data.append(seq)
        starts.append(ind_start)
        ends.append(ind_end)
    return data, starts, ends


def make_seq_data(n_samples, seq_len):
    # Boundary tasks
    data, labels = [], []
    for _ in range(n_samples):
        input = np.random.permutation(range(seq_len)).tolist()
        target = sorted(range(len(input)), key=lambda k: input[k])
        data.append(input)
        labels.append(target)
    return data, labels
