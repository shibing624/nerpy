[![PyPI version](https://badge.fury.io/py/nerpy.svg)](https://badge.fury.io/py/nerpy)
[![Downloads](https://pepy.tech/badge/nerpy)](https://pepy.tech/project/nerpy)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/nerpy.svg)](https://github.com/shibing624/nerpy/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/nerpy.svg)](https://github.com/shibing624/nerpy/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

# NERpy
🌈 Implementation of Named Entity Recognition using Python. 

**nerpy**实现了Bert2Tag、Bert2Span等多种命名实体识别模型，并在标准数据集上比较了各模型的效果。


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

# Evaluation

### 实体识别

- 英文实体识别数据集的评测结果：

| Arch | Backbone | Model Name | CoNLL-2003 | 
| :-- | :--- | :--- | :-: |
| BertSoftmax | bert-base-uncased | bert-softmax-base-uncased | - |

- 中文实体识别数据集的评测结果：

| Arch | Backbone | Model Name | CNER | PEOPLE | Avg | QPS |
| :-- | :--- | :--- | :-: | :-: | :-: | :-: |
| BertSoftmax | bert-base-chinese | bert4ner-base-chinese | 94.98 | 95.25 | 95.12 | - |

- 本项目release模型的中文匹配评测结果：

| Arch | Backbone | Model Name | CNER | PEOPLE | Avg | QPS |
| :-- | :--- | :---- | :-: | :-: | :-: | :-: |
| BertSoftmax | bert-base-chinese | shibing624/bert4ner-base-chinese | 94.98 | 95.25 | 95.12 | - |

说明：
- 结果值均使用F1
- 结果均只用该数据集的train训练，在test上评估得到的表现，没用外部数据
- `shibing624/bert4ner-base-chinese`模型达到同级别参数量SOTA效果，是用BertSoftmax方法训练，
 运行[examples/training_ner_model_file_demo.py](examples/training_ner_model_file_demo.py)代码可在各数据集复现结果
- 各预训练模型均可以通过transformers调用，如中文BERT模型：`--model_name bert-base-chinese`
- 中文实体识别数据集下载[链接见下方](#数据集)
- QPS的GPU测试环境是Tesla V100，显存32GB

# Demo

Official Demo: http://42.193.145.218/product/short_text_sim/

HuggingFace Demo: https://huggingface.co/spaces/shibing624/nerpy

![](docs/hf.png)

# Install
```
pip3 install torch # conda install pytorch
pip3 install -U nerpy
```

or

```
git clone https://github.com/shibing624/nerpy.git
cd nerpy
python3 setup.py install
```


# Usage

## 命名实体识别

基于以上`fine-tuned model`识别实体：

```shell
>>> from nerpy import NERModel
>>> model = NERModel("bert", "shibing624/bert4ner-base-chinese")
>>> predictions, raw_outputs, entities = model.predict(["常建良，男，1963年出生，工科学士，高级工程师"], split_on_space=False)
entities: [('常建良', 'NAME'), ('工科', 'PRO'), ('学士', 'EDU'), ('高级工程师', 'TITLE')]
```

example: [examples/base_zh_demo.py](examples/base_zh_demo.py)

```python
import sys

sys.path.append('..')
from nerpy import NERModel

if __name__ == '__main__':
    # 中文实体识别模型(BertSoftmax): shibing624/bert4ner-base-chinese
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
[('常建良', 'NAME'), ('工科', 'PRO'), ('学士', 'EDU'), ('高级工程师', 'TITLE'), ('北京物资学院', 'ORG'), ('客座副教授', 'TITLE')]
[('国家物资局', 'ORG'), ('物资部', 'ORG'), ('国内贸易部金属材料流通司', 'ORG'), ('科员', 'TITLE'), ('主任科员', 'TITLE')]
```

- `shibing624/bert4ner-base-chinese`模型是BertSoftmax方法在中文CNER数据集训练得到的，模型已经上传到huggingface的
模型库[shibing624/bert4ner-base-chinese](https://huggingface.co/shibing624/bert4ner-base-chinese)，
是`nerpy.NERModel`指定的默认模型，可以通过上面示例调用，或者如下所示用[transformers库](https://github.com/huggingface/transformers)调用，
模型自动下载到本机路径：`~/.cache/huggingface/transformers`

#### Usage (HuggingFace Transformers) (doing)
Without [nerpy](https://github.com/shibing624/nerpy), you can use the model like this: 

First, you pass your input through the transformer model, then you have to apply the bio to get the entity words.

example: [examples/use_origin_transformers_demo.py](examples/use_origin_transformers_demo.py)

```python
import os
import torch
from transformers import AutoTokenizer, AutoModel
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('shibing624/bert4ner-base-chinese')
model = AutoModel.from_pretrained('shibing624/bert4ner-base-chinese')
sentences = ['常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授',
             '在国家物资局、物资部、国内贸易部金属材料流通司从事调拨分配工作']
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

entities = model_output
print("Sentence entity:")
print(entities)
```


### 数据集

#### 中文实体识别数据集


| 数据集 | 语料 | 下载链接 | 文件大小 |
| :------- | :--------- | :---------: | :---------: |
| **`CNER中文实体识别数据集`** | CNER(12万字) | [CNER github](https://github.com/shibing624/nerpy/tree/main/examples/data/cner)| 1.1MB |
| **`PEOPLE中文实体识别数据集`** | 人民日报实体集（200万字） | [PEOPLE github](https://github.com/shibing624/nerpy/tree/main/examples/data/people)| 12.8MB |

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


<img src="docs/bert.png" width="300" />


模型文件组成：
```
shibing624/bert4ner-base-chinese
    ├── config.json
    ├── model_args.json
    ├── eval_result.txt
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.txt
```

#### BertSoftmax 模型训练和预测
- 在中文CNER数据集训练和评估`BertSoftmax`模型

example: [examples/training_ner_model_file_demo.py](examples/training_ner_model_file_demo.py)

```shell
cd examples
python3 training_ner_model_file_demo.py --do_train --do_predict --num_epochs 5
```
- 在英文CoNLL-2003数据集训练和评估`BertSoftmax`模型

example: [examples/training_ner_model_file_demo.py](examples/training_ner_model_file_demo.py)

```shell
cd examples
python3 training_ner_model_file_demo.py --do_train --do_predict --num_epochs 5
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
- [transformers](https://github.com/huggingface/transformers)
