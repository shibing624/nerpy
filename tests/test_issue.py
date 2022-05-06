# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import pytest

sys.path.append('..')

from nerpy.ner_model import NERModel


def test_diff(self):
    a = '研究团队面向国家重大战略需求追踪国际前沿发展借鉴国际人工智能研究领域的科研模式有效整合创新资源解决复'
    b = '英汉互译比较语言学'
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
    print(model)
    assert model is not None
