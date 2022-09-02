# Dataset

## CNER
中文CNER实体识别数据集
### input format

Input format (prefer BIO tag scheme), with each character its label for one line. Sentences are splited with a null line.

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

### CNER result

The overall performance of BERT on **test**:

|              | Accuracy  | Recall    | F1  |
| ------------ | ------------------ | ------------------ | ------------------ |
| BERT+Softmax | 0.9395     | 0.9604     | 0.9498     |
| BERT+CRF     | 0.9539     | **0.9644** | 0.9591     |
| BERT+Span    | **0.9620** | 0.9632     | **0.9626** |

## PEOPLE
中文PEOPLE（人民日报）实体识别数据集
### input format
Input format (prefer BIO tag scheme), with each character its label for one line. Sentences are splited with a null line.
```shell
新	O
世	O
纪	O
—	O
—	O
一	B-TIME
九	I-TIME
九	I-TIME
八	I-TIME
年	I-TIME
新	B-TIME
年	I-TIME
讲	O
话	O
```

### PEOPLE result

The overall performance of BERT on **test**:

|              | Accuracy  | Recall    | F1  |
| ------------ | ------------------ | ------------------ | ------------------ |
| BERT+Softmax | 0.9425     | 0.9627   | 0.9525     |

## CoNLL03
英文CoNLL-2003实体识别数据集

CoNLL-2003 dataset: [Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition](https://paperswithcode.com/dataset/conll-2003)


data set is from [here](https://github.com/hankcs/HanLP/blob/doc-zh/hanlp/datasets/ner/conll03.py)
```python
CONLL03_EN_TRAIN = 'https://file.hankcs.com/corpus/conll03_en_iobes.zip#eng.train.tsv'
'''Training set of CoNLL03 (:cite:`tjong-kim-sang-de-meulder-2003-introduction`)'''
CONLL03_EN_DEV = 'https://file.hankcs.com/corpus/conll03_en_iobes.zip#eng.dev.tsv'
'''Dev set of CoNLL03 (:cite:`tjong-kim-sang-de-meulder-2003-introduction`)'''
CONLL03_EN_TEST = 'https://file.hankcs.com/corpus/conll03_en_iobes.zip#eng.test.tsv'
'''Test set of CoNLL03 (:cite:`tjong-kim-sang-de-meulder-2003-introduction`)'''
```

### input format

Input format (prefer BIOES tag scheme), with each character its label for one line. Sentences are splited with a null line.

```text
EU	S-ORG
rejects	O
German	S-MISC
call	O
to	O
boycott	O
British	S-MISC
lamb	O
.	O

Peter	B-PER
Blackburn	E-PER
```

### CoNLL03 result

The overall performance of BertSoftmax on **test**:

|              | Accuracy  | Recall    | F1  |
| ------------ | ------------------ | ------------------ | ------------------ |
| BERT+Softmax | 0.8956     | 0.9132   | 0.9043     |


NER on CoNLL 2003(English) Leaderboard: https://paperswithcode.com/sota/named-entity-recognition-ner-on-conll-2003