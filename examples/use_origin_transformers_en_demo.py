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
tokenizer = AutoTokenizer.from_pretrained("shibing624/bert4ner-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("shibing624/bert4ner-base-uncased")
label_list = ["E-ORG", "E-LOC", "S-MISC", "I-MISC", "S-PER", "E-PER", "B-MISC", "O", "S-LOC",
              "E-MISC", "B-ORG", "S-ORG", "I-ORG", "B-LOC", "I-LOC", "B-PER", "I-PER"]

sentence = "AL-AIN, United Arab Emirates 1996-12-06"


def get_entity(sentence):
    tokens = tokenizer.tokenize(sentence)
    inputs = tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs).logits
    predictions = torch.argmax(outputs, dim=2)
    word_tags = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy()[1:-1])]
    print(sentence)
    print(word_tags)

    pred_labels = [i[1] for i in word_tags]
    entities = []
    line_entities = get_entities(pred_labels)
    for i in line_entities:
        word = tokens[i[1]: i[2] + 1]
        entity_type = i[0]
        entities.append((word, entity_type))

    print("Sentence entity:")
    print(entities)


get_entity(sentence)
