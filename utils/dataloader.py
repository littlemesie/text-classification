# -*- coding:utf-8 -*-

"""
@date: 2023/3/1 下午6:51
@summary:
"""
import torch
import pandas as pd
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """处理多分类"""
    def __init__(self, filepath, label_dict):
        super(TextDataset, self).__init__()
        self.train, self.label = self.load_data(filepath, label_dict)

    def load_data(self, path, label_dic):
        train = pd.read_csv(path)
        texts = train.text.to_list()
        labels = train.label.map(label_dic).to_list()
        return texts, labels

    def __len__(self):
        return len(self.train)

    def __getitem__(self, item):
        text = self.train[item]
        label = self.label[item]
        return text, label


class BatchTextDataset(object):
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        batch_text = [item[0] for item in batch]
        batch_label = [item[1] for item in batch]

        batch_token, batch_segment, batch_mask = list(), list(), list()
        for text in batch_text:
            if len(text) > self.max_len - 2:
                text = text[:self.max_len - 2]
            token = self.tokenizer.tokenize(text)
            token = ['[CLS]'] + token + ['[SEP]']
            token_id = self.tokenizer.convert_tokens_to_ids(token)

            padding = [0] * (self.max_len - len(token_id))
            token_id = token_id + padding
            batch_token.append(token_id)

        batch_tensor_token = torch.tensor(batch_token)
        batch_tensor_label = torch.tensor(batch_label)
        return batch_tensor_token, batch_tensor_label

class TextDatasetMulti(Dataset):
    """多分类标签"""
    def __init__(self, filepath):
        super(TextDatasetMulti, self).__init__()
        self.train, self.label = self.load_data(filepath)

    def load_data(self, path):
        train = pd.read_csv(path)
        train['label'] = train['label'].apply(lambda x: eval(x))
        texts = train.text.to_list()
        labels = train.label.to_list()
        return texts, labels

    def __len__(self):
        return len(self.train)

    def __getitem__(self, item):
        text = self.train[item]
        label = self.label[item]
        return text, label