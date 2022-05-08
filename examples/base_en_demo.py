# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This basic example loads a pre-trained model from the web and uses it get entities.
"""

import sys
from scipy.special import softmax
import numpy as np

sys.path.append('..')
from nerpy import NERModel

if __name__ == '__main__':
    # 英文实体识别模型(BertSoftmax): shibing624/bert4ner-base-uncased
    model = NERModel("bert", "shibing624/bert4ner-base-uncased")
    sentences = [
        "AL-AIN, United Arab Emirates 1996-12-06",
        "The former Soviet republic was playing in an Asian Cup finals tie for the first time.",
    ]
    # English text split by space, set split_on_space=True
    predictions, raw_outputs, entities = model.predict(sentences, split_on_space=True)
    print(predictions, entities)

    # More detailed predictions
    for n, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
        print("\n___________________________")
        print("Sentence: ", sentences[n])
        print("Entity: ", entities[n])
        for pred, out in zip(preds, outs):
            key = list(pred.keys())[0]
            new_out = out[key]
            preds = list(softmax(np.mean(new_out, axis=0)))
            print(key, pred[key], preds[np.argmax(preds)], preds)
