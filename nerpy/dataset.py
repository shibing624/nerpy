# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""


def load_data(file_path):
    data = []
    labels = set()
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            terms = line.split(' ')
            if len(terms) == 2:
                data.append([count, terms[0], terms[1]])
                labels.add(terms[1])
            else:
                count += 1
    return data, list(labels)
