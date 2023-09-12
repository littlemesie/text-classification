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

class SimCSEDataset(Dataset):
    def __init__(self, filepath):
        """
        Args:
            filepath: 文件路径
        """
        super(SimCSEDataset, self).__init__()
        self.texts, self.categories, self.label = self.load_data(filepath)

    def load_data(self, path):
        train = pd.read_csv(path, error_bad_lines=False)
        texts = train.text.to_list()
        categories = train.category.to_list()
        labels = train.label.to_list()
        return texts, categories, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        category = self.categories[item]
        label = self.label[item]
        return text, category, label

class BatchSimCSEDataset:
    """
    call function for tokenizing and getting batch text
    """
    def __init__(self, tokenizer, max_len=312):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def text2id(self, batch_text):
        return self.tokenizer(batch_text, max_length=self.max_len,
                              truncation=True, padding='max_length', return_tensors='pt')

    def __call__(self, batch):
        batch_text = [item[0] for item in batch]
        batch_category = [item[1] for item in batch]
        batch_label = [item[2] for item in batch]
        label = torch.tensor(batch_label)
        text_source = self.text2id(batch_text)
        category_source = self.text2id(batch_category)
        batch_data = {
            'text_token': text_source.get('input_ids').squeeze(1),
            'text_mask': text_source.get('attention_mask').squeeze(1),
            'text_segment': text_source.get('token_type_ids').squeeze(1),
            'category_token': category_source.get('input_ids').squeeze(1),
            'category_mask': category_source.get('attention_mask').squeeze(1),
            'category_segment': category_source.get('token_type_ids').squeeze(1),
            'label': label
        }
        return batch_data
