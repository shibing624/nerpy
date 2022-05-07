# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import torch
from transformers import AutoTokenizer, AutoModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('shibing624/bert4ner-base-chinese')
model = AutoModel.from_pretrained('shibing624/bert4ner-base-chinese')
sentences = ['常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授',
             '在国家物资局、物资部、国内贸易部金属材料流通司从事调拨分配工作']
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

entities = model_output
print("Sentence entity:")
print(entities)
