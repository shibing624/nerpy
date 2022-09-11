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


class QPSPredictTestCase(unittest.TestCase):
    def test_bertsoftmax_speed(self):
        """测试bertsoftmax_speed"""
        model = NERModel("bert", "shibing624/bert4ner-base-chinese")
        t1 = time.time()
        sentences = [
                        "常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授",
                        "1985年8月-1993年在国家物资局、物资部、国内贸易部金属材料流通司从事国家统配钢材中特种钢材品种的调拨分配工作，先后任科员、主任科员。"
                    ] * 5
        predictions, raw_outputs, entities = model.predict(sentences, split_on_space=False)
        print(predictions, entities)
        spend_time = time.time() - t1
        print(f'spend time: {spend_time}, count: {len(sentences)}')
        print('bertsoftmax qps:', len(sentences) / spend_time)
        self.assertTrue(spend_time > 0)

    def test_bertspan_speed(self):
        """测试bertspan_speed"""
        model = NERModel("bertspan", "shibing624/bertspan4ner-base-chinese")
        t1 = time.time()
        sentences = [
                        "常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授",
                        "1985年8月-1993年在国家物资局、物资部、国内贸易部金属材料流通司从事国家统配钢材中特种钢材品种的调拨分配工作，先后任科员、主任科员。"
                    ] * 5
        predictions, raw_outputs, entities = model.predict(sentences, split_on_space=False)
        print(predictions, entities)
        spend_time = time.time() - t1
        print(f'spend time: {spend_time}, count: {len(sentences)}')
        print('bertspan qps:', len(sentences) / spend_time)
        self.assertTrue(spend_time > 0)


if __name__ == '__main__':
    unittest.main()
