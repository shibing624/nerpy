# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.append('..')
from nerpy.ner_model import NERModel
from nerpy.dataset import load_data


def main():
    train_samples, train_labels = load_data('data/cner/train.char.bmes')
    eval_samples, _ = load_data('data/cner/dev.char.bmes')
    test_samples, _ = load_data('data/cner/test.char.bmes')
    train_data = pd.DataFrame(train_samples, columns=["sentence_id", "words", "labels"])
    eval_data = pd.DataFrame(eval_samples, columns=["sentence_id", "words", "labels"])
    test_data = pd.DataFrame(test_samples, columns=["sentence_id", "words", "labels"])
    print(train_data.head())
    print("train shape:", train_data.shape, " eval shape:", eval_data.shape, " test shape:", test_data.shape)

    # Create a NERModel
    model = NERModel(
        "bert",
        "bert-base-chinese",
        labels=train_labels,
        args={"overwrite_output_dir": True,
              "reprocess_input_data": True,
              "output_dir": "./output/",
              "max_seq_length": 128,
              "num_train_epochs": 3,
              "train_batch_size": 32,
              },
        use_cuda=False
    )

    # Train the model
    model.train_model(train_data, eval_data=eval_data)

    # Evaluate the model with test data
    result, model_outputs, predictions = model.eval_model(test_data)
    print(result, model_outputs, predictions)

    # Predictions on arbitary text strings
    sentences = [
        "常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授",
        "1985年8月-1993年在国家物资局、物资部、国内贸易部金属材料流通司从事国家统配钢材中特种钢材品种的调拨分配工作，先后任科员、主任科员。"
    ]
    predictions, raw_outputs = model.predict(sentences)
    print(predictions, raw_outputs)

    # More detailed preditctions
    for n, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
        print("\n___________________________")
        print("Sentence: ", sentences[n])
        for pred, out in zip(preds, outs):
            key = list(pred.keys())[0]
            new_out = out[key]
            preds = list(softmax(np.mean(new_out, axis=0)))
            print(key, pred[key], preds[np.argmax(preds)], preds)


if __name__ == '__main__':
    main()
