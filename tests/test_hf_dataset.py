# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import unittest

sys.path.append('..')

from datasets import load_dataset


class DatasetTestCase(unittest.TestCase):

    def test_data_diff(self):
        # Predict embeddings
        srcs = []
        trgs = []
        labels = []
        for term in range(100):
            srcs.append(0)
            trgs.append(1)
            labels.append(2)
            if term > 100:
                break
        print(f'{srcs[0]}')


if __name__ == '__main__':
    unittest.main()
