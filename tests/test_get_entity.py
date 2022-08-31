# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')

from nerpy.ner_model import NERModel


def test_get_entity():
    from seqeval.metrics.sequence_labeling import get_entities
    seq = ['B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O']
    r = get_entities(seq, suffix=False)
    print(r)
    # [('PER', 0, 2), ('LOC', 4, 5)]
    assert r == [('PER', 0, 2), ('LOC', 4, 5)]
    sent = ['特朗普来北京开会']
    line_entities = get_entities(seq)
    pairs = []
    for i in line_entities:
        word = sent[0][i[1]: i[2] + 1]
        entity_type = i[0]
        pairs.append((word, entity_type))
    print(pairs)
    assert pairs == [('特朗普', 'PER'), ('北京', 'LOC')]


def test_my_get_entity():
    from nerpy.ner_utils import my_get_entities as get_entities
    seq = ['B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O']
    r = get_entities(seq)
    print(r)
    # [['PER', 0, 2], ['LOC', 4, 5]]
    assert r == [['PER', 0, 2], ['LOC', 4, 5]]
    sent = ['特朗普来北京开会']
    line_entities = get_entities(seq)
    pairs = []
    for i in line_entities:
        word = sent[0][i[1]: i[2] + 1]
        entity_type = i[0]
        pairs.append((word, entity_type))
    print(pairs)
    assert pairs == [('特朗普', 'PER'), ('北京', 'LOC')]
