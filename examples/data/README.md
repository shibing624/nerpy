
### input format

Input format (prefer BIOS tag scheme), with each character its label for one line. Sentences are splited with a null line.

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

### Cner result

The overall performance of BERT on **test**:

|              | Accuracy  | Recall    | F1  |
| ------------ | ------------------ | ------------------ | ------------------ |
| BERT+Softmax | 0.9566     | 0.9613     | 0.9590     |
| BERT+CRF     | 0.9539     | **0.9644** | 0.9591     |
| BERT+Span    | **0.9620** | 0.9632     | **0.9626** |
