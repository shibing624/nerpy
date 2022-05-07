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
    train_samples, train_labels = load_data('data/people/train.char.bio.tsv')
    eval_samples, _ = load_data('data/people/dev.char.bio.tsv')
    test_samples, _ = load_data('data/people/test.char.bio.tsv')
    train_data = pd.DataFrame(train_samples, columns=["sentence_id", "words", "labels"])
    eval_data = pd.DataFrame(eval_samples, columns=["sentence_id", "words", "labels"])
    test_data = pd.DataFrame(test_samples, columns=["sentence_id", "words", "labels"])
    print('train data:', train_data.head(20))
    print('labels:', train_labels)
    print("train shape:", train_data.shape, " eval shape:", eval_data.shape, " test shape:", test_data.shape)

    # Create a NERModel
    model = NERModel(
        "bert",
        "bert-base-chinese",
        labels=train_labels,
        args={"overwrite_output_dir": True,
              "reprocess_input_data": True,
              "output_dir": "./outputs-people/",
              "max_seq_length": 128,
              "num_train_epochs": 4,
              "train_batch_size": 32,
              "classification_report": True,
              "evaluate_during_training": True,
              "evaluate_during_training_steps": 500,
              },
        use_cuda=True,
    )

    # Train the model
    model.train_model(train_data, eval_data=eval_data)

    # Evaluate the model with test data
    result, model_outputs, predictions = model.eval_model(test_data)
    print(result)

    # Predictions on text strings
    sentences = [
        "前不久，报纸上刊登了一个新闻照片，是美国总统拜登到墨尔本访问了福特澳大利亚公司。",
        "常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授",
        "1985年8月-1993年在国家物资局、物资部、国内贸易部金属材料流通司从事国家统配钢材中特种钢材品种的调拨分配工作，先后任科员、主任科员。"
    ]
    sentence_with_space = [" ".join(sentence) for sentence in sentences]
    predictions, raw_outputs, entities = model.predict(sentence_with_space, split_on_space=True)
    print(predictions, entities)

    # More detailed predictions
    for n, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
        print("\n___________________________")
        print("Sentence: ", sentences[n])
        for pred, out in zip(preds, outs):
            key = list(pred.keys())[0]
            preds = list(softmax(np.mean(out[key], axis=0)))
            print(key, pred[key], preds[np.argmax(preds)])


if __name__ == '__main__':
    main()
