# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This basic example loads a pre-trained model from the web and uses it get entities.
"""

import sys

sys.path.append('..')
from nerpy import NERModel
from nerpy.dataset import generate_tsv_vertical_bio, generate_tsv_horizontal_bio

if __name__ == '__main__':
    model = NERModel("bert", "shibing624/bert4ner-base-chinese")
    sentences = [
        "常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授",
        "1985年8月-1993年在国家物资局、物资部、国内贸易部金属材料流通司从事国家统配钢材中特种钢材品种的调拨分配工作，先后任科员、主任科员。"
    ]
    # set split_on_space=False if you use Chinese text
    predictions, raw_outputs, entities = model.predict(sentences, split_on_space=False)
    print(entities)

    entity_file = "sentence_entities.txt"
    with open(entity_file, 'w', encoding='utf-8') as f:
        for line, line_entities in zip(sentences, entities):
            ents = [i[0] for i in line_entities]
            f.write(line + '\t' + ','.join(ents) + '\n')

    # predict_file format: sentence '\t' brand1,brand2
    horizontal_file = 'hor_train.txt'
    vertical_file = 'ver_train.txt'
    # to hor bio
    generate_tsv_horizontal_bio(entity_file, horizontal_file)
    # to vertical bio
    generate_tsv_vertical_bio(entity_file, vertical_file)
