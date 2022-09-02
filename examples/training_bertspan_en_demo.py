# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys
import pandas as pd
from loguru import logger
import time

sys.path.append('..')
from nerpy.ner_model import NERModel
from nerpy.dataset import load_data


def main():
    parser = argparse.ArgumentParser('NER task')
    parser.add_argument('--task_name', default='conll03', const='conll03', nargs='?',
                        choices=['conll03'], help='task name of dataset')
    parser.add_argument('--model_type', default='bertspan', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='bert-base-cased', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs/conll03_bertspan/', type=str, help='Model output directory')
    parser.add_argument('--best_model_dir', default='./outputs/conll03_bertspan/best_model/', type=str,
                        help='Model output directory')
    parser.add_argument('--max_seq_length', default=128, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        train_samples, train_labels = load_data(f'data/{args.task_name}/train.word.bioes.tsv')
        eval_samples, _ = load_data(f'data/{args.task_name}/dev.word.bioes.tsv')
        train_data = pd.DataFrame(train_samples, columns=["sentence_id", "words", "labels"])
        eval_data = pd.DataFrame(eval_samples, columns=["sentence_id", "words", "labels"])

        logger.info(f'train data: {train_data.head(20)}')
        logger.info(f'train labels: {train_labels}')
        logger.info(f'train shape: {train_data.shape}, eval shape: {eval_data.shape}')

        # Create a NERModel
        model = NERModel(
            args.model_type,
            args.model_name,
            labels=train_labels,
            args={"overwrite_output_dir": True,
                  "reprocess_input_data": True,
                  "output_dir": args.output_dir,
                  "best_model_dir": args.best_model_dir,
                  "max_seq_length": args.max_seq_length,
                  "num_train_epochs": args.num_epochs,
                  "train_batch_size": args.batch_size,
                  "classification_report": True,
                  "evaluate_during_training": True,
                  "do_lower_case": False,
                  },
        )
        # Train the model
        model.train_model(train_data, eval_data=eval_data)

    if args.do_predict:
        test_samples, _ = load_data(f'data/{args.task_name}/test.word.bioes.tsv')
        test_data = pd.DataFrame(test_samples, columns=["sentence_id", "words", "labels"])
        logger.info(f'test shape: {test_data.shape}')

        model = NERModel(
            args.model_type,
            args.best_model_dir,
        )
        # Evaluate the model with test data
        t1 = time.time()
        result, model_outputs, predictions = model.eval_model(test_data)
        print(result)
        spend_time = time.time() - t1
        count = len(test_data['sentence_id'].unique())
        print('spend time:', spend_time, ' sentences size:', count, ' qps:', count / spend_time)

        # Predictions on text strings
        sentences = [
            "AL-AIN, United Arab Emirates 1996-12-06",
            "The former Soviet republic was playing in an Asian Cup finals tie for the first time.",
        ]
        predictions, raw_outputs, entities = model.predict(sentences, split_on_space=True)
        print(predictions, entities)


if __name__ == '__main__':
    main()
