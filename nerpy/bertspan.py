# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: BertSpan Model
"""
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from torch.utils.data import Dataset
from seqeval.metrics.sequence_labeling import get_entities
from nerpy.ner_utils import InputExample


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj


class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x


class BertSpanForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, use_soft_label=True):
        super(BertSpanForTokenClassification, self).__init__(config)
        self.soft_label = use_soft_label
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = ((start_logits, end_logits),) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs


def get_span_subject(start_ids, end_ids, input_lens=None):
    """Get span entities from start and end ids."""
    subjects = []
    for k in range(len(start_ids)):
        subject = []
        m = start_ids[k][1: -1]
        n = end_ids[k][1: -1]
        if input_lens is not None:
            l = input_lens[k] - 2
        else:
            l = None
        for i, s_l in enumerate(m[:l]):
            if s_l == 0:
                continue
            for j, e_l in enumerate(n[i:l]):
                if s_l == e_l:
                    subject.append((s_l, i, i + j))
                    break
        subjects.append(subject)
    return subjects


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, start_ids=None, end_ids=None, input_len=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_ids = start_ids
        self.end_ids = end_ids
        self.input_len = input_len


def convert_example_to_feature(
        example,
        label_map,
        max_seq_length,
        tokenizer,
        cls_token_at_end,
        cls_token,
        cls_token_segment_id,
        sep_token,
        sep_token_extra,
        pad_on_left,
        pad_token,
        pad_token_segment_id,
        pad_token_label_id,
        sequence_a_segment_id,
        mask_padding_with_zero,
        return_input_feature=True,
):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    start_label_map = {f'B-{k}': v for k, v in label_map.items()}
    end_label_map = {f'I-{k}': v for k, v in label_map.items()}
    tokens = []
    start_ids = []
    end_ids = []
    for i, (word, label) in enumerate(zip(example.words, example.labels)):
        if example.tokenized_word_ids is None:
            word_tokens = tokenizer.tokenize(word)
        else:
            word_tokens = example.tokenized_word_ids[i]
        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        if word_tokens:  # avoid non printable character like '\u200e' which are tokenized as a void token ''
            tokens.extend(word_tokens)
        else:
            word_tokens = tokenizer.tokenize(tokenizer.unk_token)
            tokens.extend(word_tokens)
        start_ids.extend(
            [start_label_map.get(label, pad_token_label_id)] + [pad_token_label_id] * (len(word_tokens) - 1)
        )
        end_ids.extend(
            [pad_token_label_id] * (len(word_tokens) - 1) + [end_label_map.get(label, pad_token_label_id)]
        )
    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        start_ids = start_ids[:(max_seq_length - special_tokens_count)]
        end_ids = end_ids[:(max_seq_length - special_tokens_count)]
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens += [sep_token]
    start_ids += [pad_token_label_id]
    end_ids += [pad_token_label_id]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        start_ids += [pad_token_label_id]
        end_ids += [pad_token_label_id]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        tokens += [cls_token]
        start_ids += [pad_token_label_id]
        end_ids += [pad_token_label_id]
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        start_ids = [pad_token_label_id] + start_ids
        end_ids = [pad_token_label_id] + end_ids
        segment_ids = [cls_token_segment_id] + segment_ids

    if example.tokenized_word_ids is None:
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
    else:
        input_ids = tokens

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    input_len = len(input_ids)
    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        start_ids = ([pad_token_label_id] * padding_length) + start_ids
        end_ids = ([pad_token_label_id] * padding_length) + end_ids
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        start_ids += [pad_token_label_id] * padding_length
        end_ids += [pad_token_label_id] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(start_ids) == max_seq_length
    assert len(end_ids) == max_seq_length

    if return_input_feature:
        return InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_ids=start_ids,
            end_ids=end_ids,
            input_len=input_len,
        )
    else:
        return (
            input_ids,
            input_mask,
            segment_ids,
            start_ids,
            end_ids,
            input_len,
        )


def read_examples_from_file(data_file, mode):
    file_path = data_file
    guid_index = 1
    examples = []
    with open(file_path, "r", encoding="utf8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=words,
                            labels=labels,
                        )
                    )
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(
                InputExample(
                    guid="%s-%d".format(mode, guid_index),
                    words=words,
                    labels=labels,
                )
            )
    return examples


def get_examples_from_df(data):
    return [
        InputExample(
            guid=sentence_id,
            words=sentence_df["words"].tolist(),
            labels=sentence_df["labels"].tolist(),
        )
        for sentence_id, sentence_df in data.groupby(["sentence_id"])
    ]


class BertSpanDataset(Dataset):
    """BertSpan dataset, use it by dataset_class from train args"""

    def __init__(self, data, tokenizer, args, mode='train', to_predict=None, **kwargs):
        if data is None and to_predict:
            self.example_lines = to_predict
        else:
            if isinstance(data, str):
                self.example_lines = read_examples_from_file(data, mode)
            else:
                self.example_lines = get_examples_from_df(data)
        self.tokenizer = tokenizer
        self.args = args
        self.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        self.label_map = {label: i for i, label in enumerate(self.args.labels_list)}
        self.mode = mode

    def __getitem__(self, idx):
        example = self.example_lines[idx]
        features = convert_example_to_feature(
            example,
            label_map=self.label_map,
            max_seq_length=self.args.max_seq_length,
            tokenizer=self.tokenizer,
            cls_token_at_end=False,
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=self.pad_token_id,
            pad_token_segment_id=0,
            pad_token_label_id=self.pad_token_id,
            sequence_a_segment_id=0,
            mask_padding_with_zero=True,
            return_input_feature=True
        )
        all_input_ids = torch.tensor(features.input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(features.input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(features.segment_ids, dtype=torch.long)
        all_start_ids = torch.tensor(features.start_ids, dtype=torch.long)
        all_end_ids = torch.tensor(features.end_ids, dtype=torch.long)
        all_input_len = torch.tensor(features.input_len, dtype=torch.long)
        return (all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_input_len)

    def __len__(self):
        return len(self.example_lines)


class SpanEntityScore:
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([self.id2label[x[0]] for x in self.origins])
        found_counter = Counter([self.id2label[x[0]] for x in self.founds])
        right_counter = Counter([self.id2label[x[0]] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend([pre_entity for pre_entity in pred_subject if pre_entity in true_subject])


def check_span_labels(labels):
    """
    check span labels don't has '-'
    :param labels:
    :return:
    """
    if not labels:
        return False
    for label in labels:
        if '-' in label:
            return False
    return True
