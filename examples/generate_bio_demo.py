# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This basic example predict with fine-tuned model and uses it generate entities samples.
"""

import sys

sys.path.append('..')
from nerpy import NERModel
from nerpy.dataset import generate_tsv_vertical_bio, generate_tsv_horizontal_bio


def save_bio(file_path, bio_tags):
    with open(file_path, 'w', encoding='utf-8') as fw:
        for sent_tag in bio_tags:
            fw.write(sent_tag + '\n')


if __name__ == '__main__':
    model = NERModel("bert", "shibing624/bert4ner-base-chinese")
    sentences = [
        "常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授",
        "1985年8月-1993年在国家物资局、物资部、国内贸易部金属材料流通司从事国家统配钢材中特种钢材品种的调拨分配工作，先后任科员、主任科员。"
    ]
    # set split_on_space=False if you use Chinese text
    predictions, raw_outputs, entities = model.predict(sentences, split_on_space=False)
    print(entities)

    entity_file = "sentence_entities.tsv"
    sentence_entities = []
    with open(entity_file, 'w', encoding='utf-8') as f:
        for line, line_entities in zip(sentences, entities):
            ents = [i[0] for i in line_entities]
            f.write(line + '\t' + ','.join(ents) + '\n')
            sentence_entities.append(line + '\t' + ','.join(ents))

    # predict_file format: sentence '\t' brand1,brand2
    horizontal_file = 'hor_train.tsv'
    vertical_file = 'ver_train.tsv'
    # to hor bio
    horizontal_bio_tags = generate_tsv_horizontal_bio(sentence_entities)
    save_bio(horizontal_file, horizontal_bio_tags)
    # to vertical bio
    vertical_bio_tags = generate_tsv_vertical_bio(sentence_entities)
    save_bio(vertical_file, vertical_bio_tags)
