[![PyPI version](https://badge.fury.io/py/nerpy.svg)](https://badge.fury.io/py/nerpy)
[![Downloads](https://pepy.tech/badge/nerpy)](https://pepy.tech/project/nerpy)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/nerpy.svg)](https://github.com/shibing624/nerpy/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.8%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/nerpy.svg)](https://github.com/shibing624/nerpy/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

# NERpy
🌈 Implementation of Named Entity Recognition using Python. 

**nerpy**实现了BertSoftmax、BertCrf、BertSpan等多种命名实体识别模型，并在标准数据集上比较了各模型的效果。


**Guide**
- [Feature](#Feature)
- [Evaluation](#Evaluation)
- [Install](#install)
- [Usage](#usage)
- [Contact](#Contact)
- [Reference](#reference)


# Feature
### 命名实体识别模型
- [BertSoftmax](nerpy/ner_model.py)：BertSoftmax基于BERT预训练模型实现实体识别，本项目基于PyTorch实现了BertSoftmax模型的训练和预测
- [BertSpan](nerpy/bertspan.py)：BertSpan基于BERT训练span边界的表示，模型结构更适配实体边界识别，本项目基于PyTorch实现了BertSpan模型的训练和预测

# Evaluation

### 实体识别

- 英文实体识别数据集的评测结果：

| Arch | Backbone | Model Name | CoNLL-2003 | QPS |
| :-- | :--- | :--- | :-: | :--: |
| BertSoftmax | bert-base-uncased | bert4ner-base-uncased | 90.43 | 235 | 
| BertSoftmax | bert-base-cased | bert4ner-base-cased | 91.17 | 235 | 

- 中文实体识别数据集的评测结果：

| Arch | Backbone | Model Name | CNER | PEOPLE | Avg | QPS |
| :-- | :--- | :--- | :-: | :-: | :-: | :-: |
| BertSoftmax | bert-base-chinese | bert4ner-base-chinese | 94.98 | 95.25 | 95.12 | 222 |
| BertSpan | bert-base-chinese | bertspan4ner-base-chinese | 96.03 | 96.06 | 96.04 | 254 |

- 本项目release模型的实体识别评测结果：

| Arch | Backbone | Model Name | CNER(zh) | PEOPLE(zh) | CoNLL-2003(en) | QPS |
| :-- | :--- | :---- | :-: | :-: | :-: | :-: |
| BertSpan | bert-base-chinese | shibing624/bertspan4ner-base-chinese | 96.03 | 96.06 | - | 254 |
| BertSoftmax | bert-base-chinese | shibing624/bert4ner-base-chinese | 94.98 | 95.25 | - | 222 |
| BertSoftmax | bert-base-uncased | shibing624/bert4ner-base-uncased | - | - | 90.43 | 243 |

说明：
- 结果值均使用F1
- 结果均只用该数据集的train训练，在test上评估得到的表现，没用外部数据
- `shibing624/bertspan4ner-base-chinese`模型达到base级别里SOTA效果，是用BertSpan方法训练的，
 运行[examples/training_bertspan_zh_demo.py](examples/training_bertspan_zh_demo.py)代码可在各中文数据集复现结果
- `shibing624/bert4ner-base-chinese`模型达到base级别里较好效果，是用BertSoftmax方法训练的，
 运行[examples/training_ner_model_zh_demo.py](examples/training_ner_model_zh_demo.py)代码可在各中文数据集复现结果
- `shibing624/bert4ner-base-uncased`模型是用BertSoftmax方法训练的，
 运行[examples/training_ner_model_en_demo.py](examples/training_ner_model_en_demo.py)代码可在CoNLL-2003英文数据集复现结果
- 各预训练模型均可以通过transformers调用，如中文BERT模型：`--model_name bert-base-chinese`
- 中文实体识别数据集下载[链接见下方](#数据集)
- QPS的GPU测试环境是Tesla V100，显存32GB

# Demo

Demo: https://huggingface.co/spaces/shibing624/nerpy

![](docs/hf.png)

run example: [examples/gradio_demo.py](examples/gradio_demo.py) to see the demo:
```shell
python examples/gradio_demo.py
```

 
# Install
python 3.8+

```shell
pip install torch # conda install pytorch
pip install -U nerpy
```

or

```shell
pip install torch # conda install pytorch
pip install -r requirements.txt

git clone https://github.com/shibing624/nerpy.git
cd nerpy
pip install --no-deps .
```

# Usage

## 命名实体识别

#### 英文实体识别：

```shell
>>> from nerpy import NERModel
>>> model = NERModel("bert", "shibing624/bert4ner-base-uncased")
>>> predictions, raw_outputs, entities = model.predict(["AL-AIN, United Arab Emirates 1996-12-06"], split_on_space=True)
entities:  [('AL-AIN,', 'LOC'), ('United Arab Emirates', 'LOC')]
```

#### 中文实体识别：

```shell
>>> from nerpy import NERModel
>>> model = NERModel("bert", "shibing624/bert4ner-base-chinese")
>>> predictions, raw_outputs, entities = model.predict(["常建良，男，1963年出生，工科学士，高级工程师"], split_on_space=False)
entities: [('常建良', 'PER'), ('1963年', 'TIME')]
```

example: [examples/base_zh_demo.py](examples/base_zh_demo.py)

```python
import sys

sys.path.append('..')
from nerpy import NERModel

if __name__ == '__main__':
    # BertSoftmax中文实体识别模型: NERModel("bert", "shibing624/bert4ner-base-chinese")
    # BertSpan中文实体识别模型: NERModel("bertspan", "shibing624/bertspan4ner-base-chinese")
    model = NERModel("bert", "shibing624/bert4ner-base-chinese")
    sentences = [
        "常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授",
        "1985年8月-1993年在国家物资局、物资部、国内贸易部金属材料流通司从事国家统配钢材中特种钢材品种的调拨分配工作，先后任科员、主任科员。"
    ]
    predictions, raw_outputs, entities = model.predict(sentences)
    print(entities)
```

output:
```
[('常建良', 'PER'), ('1963年', 'TIME'), ('北京物资学院', 'ORG')]
[('1985年', 'TIME'), ('8月', 'TIME'), ('1993年', 'TIME'), ('国家物资局', 'ORG'), ('物资部', 'ORG'), ('国内贸易部金属材料流通司', 'ORG')]
```

- `shibing624/bert4ner-base-chinese`模型是BertSoftmax方法在中文PEOPLE(人民日报)数据集训练得到的，模型已经上传到huggingface的
模型库[shibing624/bert4ner-base-chinese](https://huggingface.co/shibing624/bert4ner-base-chinese)，
是`nerpy.NERModel`指定的默认模型，可以通过上面示例调用，或者如下所示用[transformers库](https://github.com/huggingface/transformers)调用，
模型自动下载到本机路径：`~/.cache/huggingface/transformers`
- `shibing624/bertspan4ner-base-chinese`模型是BertSpan方法在中文PEOPLE(人民日报)数据集训练得到的，模型已经上传到huggingface的
模型库[shibing624/bertspan4ner-base-chinese](https://huggingface.co/shibing624/bertspan4ner-base-chinese)


#### Usage (HuggingFace Transformers)
Without [nerpy](https://github.com/shibing624/nerpy), you can use the model like this: 

First, you pass your input through the transformer model, then you have to apply the bio tag to get the entity words.

example: [examples/predict_use_origin_transformers_zh_demo.py](examples/predict_use_origin_transformers_zh_demo.py)

```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics.sequence_labeling import get_entities

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("shibing624/bert4ner-base-chinese")
model = AutoModelForTokenClassification.from_pretrained("shibing624/bert4ner-base-chinese")
label_list = ['I-ORG', 'B-LOC', 'O', 'B-ORG', 'I-LOC', 'I-PER', 'B-TIME', 'I-TIME', 'B-PER']

sentence = "王宏伟来自北京，是个警察，喜欢去王府井游玩儿。"


def get_entity(sentence):
    tokens = tokenizer.tokenize(sentence)
    inputs = tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs).logits
    predictions = torch.argmax(outputs, dim=2)
    char_tags = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())][1:-1]
    print(sentence)
    print(char_tags)

    pred_labels = [i[1] for i in char_tags]
    entities = []
    line_entities = get_entities(pred_labels)
    for i in line_entities:
        word = sentence[i[1]: i[2] + 1]
        entity_type = i[0]
        entities.append((word, entity_type))

    print("Sentence entity:")
    print(entities)


get_entity(sentence)
```
output:
```shell
王宏伟来自北京，是个警察，喜欢去王府井游玩儿。
[('王', 'B-PER'), ('宏', 'I-PER'), ('伟', 'I-PER'), ('来', 'O'), ('自', 'O'), ('北', 'B-LOC'), ('京', 'I-LOC'), ('，', 'O'), ('是', 'O'), ('个', 'O'), ('警', 'O'), ('察', 'O'), ('，', 'O'), ('喜', 'O'), ('欢', 'O'), ('去', 'O'), ('王', 'B-LOC'), ('府', 'I-LOC'), ('井', 'I-LOC'), ('游', 'O'), ('玩', 'O'), ('儿', 'O'), ('。', 'O')]
Sentence entity:
[('王宏伟', 'PER'), ('北京', 'LOC'), ('王府井', 'LOC')]
```

### 数据集

#### 实体识别数据集


| 数据集 | 语料 | 下载链接 | 文件大小 |
| :------- | :--------- | :---------: | :---------: |
| **`CNER中文实体识别数据集`** | CNER(12万字) | [CNER github](https://github.com/shibing624/nerpy/tree/main/examples/data/cner)| 1.1MB |
| **`PEOPLE中文实体识别数据集`** | 人民日报数据集（200万字） | [PEOPLE github](https://github.com/shibing624/nerpy/tree/main/examples/data/people)| 12.8MB |
| **`CoNLL03英文实体识别数据集`** | CoNLL-2003数据集（22万字） | [CoNLL03 github](https://github.com/shibing624/nerpy/tree/main/examples/data/conll03)| 1.7MB |

CNER中文实体识别数据集，数据格式：

```text
美	B-LOC
国	I-LOC
的	O
华	B-PER
莱	I-PER
士	I-PER

我	O
跟	O
他	O
```


## BertSoftmax 模型

BertSoftmax实体识别模型，基于BERT的标准序列标注方法：

Network structure:


<img src="docs/bert.png" width="500" />


模型文件组成：
```
shibing624/bert4ner-base-chinese
    ├── config.json
    ├── model_args.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.txt
```

#### BertSoftmax 模型训练和预测

training example: [examples/training_ner_model_toy_demo.py](examples/bert_softmax_demo.py)


```python
import sys
import pandas as pd

sys.path.append('..')
from nerpy.ner_model import NERModel


# Creating samples
train_samples = [
    [0, "HuggingFace", "B-MISC"],
    [0, "Transformers", "I-MISC"],
    [0, "started", "O"],
    [0, "with", "O"],
    [0, "text", "O"],
    [0, "classification", "B-MISC"],
    [1, "Nerpy", "B-MISC"],
    [1, "Model", "I-MISC"],
    [1, "can", "O"],
    [1, "now", "O"],
    [1, "perform", "O"],
    [1, "NER", "B-MISC"],
]
train_data = pd.DataFrame(train_samples, columns=["sentence_id", "words", "labels"])

test_samples = [
    [0, "HuggingFace", "B-MISC"],
    [0, "Transformers", "I-MISC"],
    [0, "was", "O"],
    [0, "built", "O"],
    [0, "for", "O"],
    [0, "text", "O"],
    [0, "classification", "B-MISC"],
    [1, "Nerpy", "B-MISC"],
    [1, "Model", "I-MISC"],
    [1, "then", "O"],
    [1, "expanded", "O"],
    [1, "to", "O"],
    [1, "perform", "O"],
    [1, "NER", "B-MISC"],
]
test_data = pd.DataFrame(test_samples, columns=["sentence_id", "words", "labels"])

# Create a NERModel
model = NERModel(
    "bert",
    "bert-base-uncased",
    args={"overwrite_output_dir": True, "reprocess_input_data": True, "num_train_epochs": 1},
    use_cuda=False,
)

# Train the model
model.train_model(train_data)

# Evaluate the model
result, model_outputs, predictions = model.eval_model(test_data)
print(result, model_outputs, predictions)

# Predictions on text strings
sentences = ["Nerpy Model perform sentence NER", "HuggingFace Transformers build for text"]
predictions, raw_outputs, entities = model.predict(sentences, split_on_space=True)
print(predictions, entities)
```

- 在中文CNER数据集训练和评估`BertSoftmax`模型

example: [examples/training_ner_model_zh_demo.py](examples/training_ner_model_zh_demo.py)

```shell
cd examples
python training_ner_model_zh_demo.py --do_train --do_predict --num_epochs 5 --task_name cner
```
- 在英文CoNLL-2003数据集训练和评估`BertSoftmax`模型

example: [examples/training_ner_model_en_demo.py](examples/training_ner_model_en_demo.py)

```shell
cd examples
python training_ner_model_en_demo.py --do_train --do_predict --num_epochs 5
```


#### BertSpan 模型训练和预测

- 在中文CNER数据集训练和评估`BertSpan`模型

example: [examples/training_bertspan_zh_demo.py](examples/training_bertspan_zh_demo.py)

```shell
cd examples
python training_bertspan_zh_demo.py --do_train --do_predict --num_epochs 5 --task_name cner
```

# Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/nerpy.svg)](https://github.com/shibing624/nerpy/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：
加我*微信号：xuming624, 备注：姓名-公司-NLP* 进NLP交流群。

<img src="docs/wechat.jpeg" width="200" />


# Citation

如果你在研究中使用了nerpy，请按如下格式引用：

APA:
```latex
Xu, M. nerpy: Named Entity Recognition Toolkit (Version 0.0.2) [Computer software]. https://github.com/shibing624/nerpy
```

BibTeX:
```latex
@software{Xu_nerpy_Text_to,
author = {Xu, Ming},
title = {{nerpy: Named Entity Recognition Toolkit}},
url = {https://github.com/shibing624/nerpy},
version = {0.0.2}
}
```

# License


授权协议为 [The Apache License 2.0](LICENSE)，可免费用做商业用途。请在产品说明中附加nerpy的链接和授权协议。


# Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在`tests`添加相应的单元测试
 - 使用`python -m pytest -v`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。

# Reference
- [huggingface/transformers](https://github.com/huggingface/transformers)
