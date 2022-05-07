# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install gradio
"""

import gradio as gr
from nerpy import NERModel

# 中文实体识别模型(BertSoftmax)
ner_model = NERModel(model_type='bert', model_name='shibing624/bert4ner-base-chinese')


def ai_text(sentence):
    predictions, raw_outputs, entities = ner_model.predict([sentence])
    print("{} \t Entity: {}".format(sentence, entities))

    return entities


if __name__ == '__main__':
    examples = [
        ['常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授'],
        ['在国家物资局、物资部、国内贸易部金属材料流通司从事调拨分配工作'],
    ]
    input = gr.inputs.Textbox(lines=4, placeholder="Enter Sentence")

    output_text = gr.outputs.Textbox()
    gr.Interface(ai_text,
                 inputs=[input],
                 outputs=[output_text],
                 # theme="grass",
                 title="Chinese Text to Vector Model shibing624/bert4ner-base-chinese",
                 description="Copy or input Chinese text here. Submit and the machine will calculate the NER entity.",
                 article="Link to <a href='https://github.com/shibing624/nerpy' style='color:blue;' target='_blank\'>Github REPO</a>",
                 examples=examples
                 ).launch()
