#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from bequick.utils import zip_open


def read_postag_dataset(filename):
    """
    Read raw dataset from file
    :param filename: str, the path to the file.
    :return: list(dict)
    """
    dataset = []
    fpi = zip_open(filename)
    for raw_data in fpi.read().strip().split('\n'):
        data = []
        for token in raw_data.split():
            form, pos = token.rsplit('/', 1)
            data.append({'form': form, 'pos': pos})
        dataset.append(data)
    return dataset


def read_conllx_dataset(filename, max_length=None):
    dataset = []
    fpi = zip_open(filename)
    for raw_data in fpi.read().strip().split('\n\n'):
        data = []
        lines = raw_data.split('\n')
        if max_length is not None and len(lines) > max_length:
            continue
        for line in lines:
            tokens = line.split('\t')
            data.append({
                'id': int(tokens[0]),
                'form': tokens[1],
                'pos': tokens[3],
                'head': int(tokens[6]),
                'deprel': tokens[7]
            })
        dataset.append(data)
    return dataset


def get_alphabet(dataset, keyword, init_with_default_keys=True):
    """

    :param dataset:
    :param keyword:
    :param init_with_default_keys: bool
    :return:
    """
    if init_with_default_keys:
        ret = {None: 0, 'UNK': 1}
    else:
        ret = {}
    for data in dataset:
        for item in data:
            form = item[keyword]
            if form not in ret:
                ret[form] = len(ret)
    return ret