
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

### PEOPLE result

The overall performance of BERT on **test**:

|              | Accuracy  | Recall    | F1  |
| ------------ | ------------------ | ------------------ | ------------------ |
| BERT+Softmax | 0.9425     | 0.9627   | 0.9525     |