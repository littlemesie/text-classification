# -*- coding:utf-8 -*-

"""
@date: 2023/3/1 下午6:49
@summary:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, num_kernels, kernel_size, stride=1, emb_size=128,
                 dropout=0.0, padding_index=0):
        """
        :param vocab_size: 词表大小
        :param num_classes: 类别数
        :param num_kernels: 卷积核数量(channels数)
        :param kernel_size:  卷积核尺寸
        :param stride: stride
        :param emb_size: 词向量维度
        :param dropout: dropout值
        :param padding_index: padding_index
        """
        super(TextCNN, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_index)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_kernels, (k, emb_size), stride) for k in kernel_size])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_kernels * len(kernel_size), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out