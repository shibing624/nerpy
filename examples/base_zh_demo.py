# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This basic example loads a pre-trained model from the web and uses it get entities.
"""

import sys

sys.path.append('..')
from nerpy import NERModel

if __name__ == '__main__':
    # BertSoftmax中文实体识别模型: NERModel("bert", "shibing624/bert4ner-base-chinese")
    # BertSpan中文实体识别模型: NERModel("bertspan", "shibing624/bertspan4ner-base-chinese")
    model = NERModel("bert", "shibing624/bert4ner-base-chinese")
    sentences = [
        "常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授",
        "1985年8月-1993年在国家物资局、物资部、国内贸易部金属材料流通司从事国家统配钢材中特种钢材品种的调拨分配工作，先后任科员、主任科员。"
    ]
    # set split_on_space=False if you use Chinese text
    predictions, raw_outputs, entities = model.predict(sentences, split_on_space=False)
    print(predictions, entities)

    # More detailed predictions
    for n, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
        print("\n___________________________")
        print("Sentence: ", sentences[n])
        print("Entity: ", entities[n])
