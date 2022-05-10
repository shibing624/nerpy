# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import os


def load_data(file_path):
    data = []
    labels = set()
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '-DOCSTART-' in line:
                continue
            terms = line.split()
            if len(terms) == 2:
                data.append([count, terms[0], terms[1]])
                labels.add(terms[1])
            else:
                count += 1
    return data, list(labels)


def generate_tsv_horizontal_bio(predict_result_file, horizontal_file, B='B-ORG', I='I-ORG', O='O'):
    """
    Generate tsv file with horizontal bio format.
    :param predict_result_file: format: sentence '\t' brand1,brand2
    :param horizontal_file: format: sentence '\t' O O O B I O
    :param B:
    :param I:
    :param O:
    :return: None
    """
    with open(predict_result_file, 'r', encoding='utf-8') as fr, \
            open(horizontal_file, 'w', encoding='utf-8') as fw:
        for line in fr:
            line = line.strip()
            terms = line.split("\t")
            i = terms[0]
            b = terms[1].split(',')[0]

            if not b:
                continue
            brand_idx = i.index(b)
            if brand_idx > -1:
                brand_start = brand_idx
                brand_len = len(b)
                if brand_len == 1:
                    continue
                out = ' '.join([O] * brand_start + [B] + [I] * (brand_len - 1) + [O] * (
                        len(i) - brand_start - brand_len))
            else:
                out = ' '.join([O] * len(i))
            fw.write(i + '\t' + out + '\n')


def generate_tsv_vertical_bio(predict_result_file, out_vertical_file, B='B-ORG', I='I-ORG', O='O'):
    """
    Generate tsv file with vertical bio format.
    :param predict_result_file: format: sentence '\t' brand1,brand2
    :param out_vertical_file: output file
        char '\t' O
        char '\t' B
        char '\t' I
    :param B:
    :param I:
    :param O:
    :return: None
    """
    with open(predict_result_file, 'r', encoding='utf-8') as fr, \
            open(out_vertical_file, 'w', encoding='utf-8') as fw:
        for line in fr:
            line = line.strip()
            terms = line.split("\t")
            sentence = terms[0]
            chars = list(sentence)
            b = terms[1].split(',')[0]

            if not b:
                continue
            brand_idx = sentence.index(b)
            if brand_idx > -1:
                brand_start = brand_idx
                brand_len = len(b)
                if brand_len == 1:
                    continue
                tags = [O] * brand_start + [B] + [I] * (brand_len - 1) + [O] * (
                        len(sentence) - brand_start - brand_len)
            else:
                tags = [O] * len(sentence)

            if len(chars) != len(tags):
                continue
            for i in range(len(chars)):
                fw.write(chars[i] + '\t' + tags[i] + '\n')
            fw.write('\n')
