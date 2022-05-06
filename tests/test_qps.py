# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import unittest
import time

sys.path.append('..')
from nerpy.ner_model import NERModel
from nerpy.dataset import load_data

pwd_path = os.path.abspath(os.path.dirname(__file__))
test_path = os.path.join(pwd_path, '../examples/data/cner/test.char.bmes')

model = NERModel(
    "bert",
    "bert-base-chinese",
    args={"overwrite_output_dir": True,
          "reprocess_input_data": True,
          "output_dir": "./output/",
          "max_seq_length": 128,
          "num_train_epochs": 3,
          "train_batch_size": 32,
          },
    use_cuda=False
)


class QPSEncoderTestCase(unittest.TestCase):
    def test_cosent_speed(self):
        """测试cosent_speed"""
        data, labels = load_data(test_path)
        print('sente size:', len(data))
        t1 = time.time()
        # model.predict(data)
        time.sleep(1)
        spend_time = time.time() - t1
        print('spend time:', spend_time, ' seconds')
        print('cosent_sbert qps:', len(data) / spend_time)

    def test_sbert_speed(self):
        """测试sbert_speed"""
        pass

    def test_w2v_speed(self):
        """测试w2v_speed"""
        pass


if __name__ == '__main__':
    unittest.main()
