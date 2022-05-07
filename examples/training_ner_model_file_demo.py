# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys
import numpy as np
import pandas as pd
from loguru import logger
from scipy.special import softmax

sys.path.append('..')
from nerpy.ner_model import NERModel
from nerpy.dataset import load_data


def main():
    parser = argparse.ArgumentParser('NER task')
    parser.add_argument('--task_name', default='cner', const='cner', nargs='?',
                        choices=['cner', 'people'], help='task name of dataset')
    parser.add_argument('--model_type', default='bert', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='hfl/chinese-macbert-base', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--use_cuda', action='store_true', help='Whether to use cuda.')
    parser.add_argument('--output_dir', default='./outputs/cner-model', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=128, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=4, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)

    train_samples, train_labels = load_data(f'data/{args.task_name}/train.char.bio.tsv')
    eval_samples, _ = load_data(f'data/{args.task_name}/dev.char.bio.tsv')
    test_samples, _ = load_data(f'data/{args.task_name}/test.char.bio.tsv')
    train_data = pd.DataFrame(train_samples, columns=["sentence_id", "words", "labels"])
    eval_data = pd.DataFrame(eval_samples, columns=["sentence_id", "words", "labels"])
    test_data = pd.DataFrame(test_samples, columns=["sentence_id", "words", "labels"])
    logger.info(f'train data: {train_data.head(20)}')
    logger.info(f'train labels: {train_labels}')
    logger.info(f'train shape: {train_data.shape}, eval shape: {eval_data.shape}, test shape: {test_data.shape}')

    # Create a NERModel
    model = NERModel(
        args.model_type,
        args.model_name,
        labels=train_labels,
        args={"overwrite_output_dir": True,
              "reprocess_input_data": True,
              "output_dir": args.output_dir,
              "max_seq_length": args.max_seq_length,
              "num_train_epochs": args.num_epochs,
              "train_batch_size": args.batch_size,
              "classification_report": True,
              "evaluate_during_training": True,
              },
        use_cuda=args.use_cuda,
    )
    if args.do_train:
        # Train the model
        model.train_model(train_data, eval_data=eval_data)

    if args.do_predict:
        # Evaluate the model with test data
        result, model_outputs, predictions = model.eval_model(test_data)
        print(result)

        # Predictions on text strings
        sentences = [
            "前不久，报纸上刊登了一个新闻照片，是美国总统拜登到墨尔本访问了福特澳大利亚公司。",
            "常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授",
            "1985年8月-1993年在国家物资局、物资部、国内贸易部金属材料流通司从事国家统配钢材中特种钢材品种的调拨分配工作，先后任科员、主任科员。"
        ]
        predictions, raw_outputs, entities = model.predict(sentences, split_on_space=False)
        print(predictions, entities)

        # More detailed predictions
        for n, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
            print("\n___________________________")
            print("Sentence: ", sentences[n])
            print("Entity: ", entities[n])
            for pred, out in zip(preds, outs):
                key = list(pred.keys())[0]
                preds = list(softmax(np.mean(out[key], axis=0)))
                print(key, pred[key], preds[np.argmax(preds)])


if __name__ == '__main__':
    main()
