# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics.sequence_labeling import get_entities

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("shibing624/bert4ner-base-chinese")
model = AutoModelForTokenClassification.from_pretrained("shibing624/bert4ner-base-chinese")
label_list = ['I-ORG', 'B-LOC', 'O', 'B-ORG', 'I-LOC', 'I-PER', 'B-TIME', 'I-TIME', 'B-PER']

sentence = "王宏伟来自北京，是个警察，喜欢去王府井游玩儿。"


def get_entity(sentence):
    inputs = tokenizer.encode(sentence, return_tensors="pt")
    tokens = tokenizer.tokenize(sentence)
    with torch.no_grad():
        outputs = model(inputs).logits
    predictions = torch.argmax(outputs, dim=2)
    char_tags = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())][1:-1]
    print(sentence)
    print(char_tags)

    pred_labels = [i[1] for i in char_tags]
    entities = []
    line_entities = get_entities(pred_labels)
    for i in line_entities:
        word = sentence[i[1]: i[2] + 1]
        entity_type = i[0]
        entities.append((word, entity_type))

    print("Sentence entity:")
    print(entities)


get_entity(sentence)
