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
        # x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = F.max_pool1d(x, (int(x.size(2)),)).squeeze(2)
        return x

    def forward(self, x):
        out = self.emb(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class DNN(nn.Module):
    def __init__(self, hidden_units):
        """
        DNN 层
        :param hidden_units: 隐含层
        """
        super(DNN, self).__init__()
        self.hidden_units = hidden_units
        self.fc_layers = [nn.Linear(hidden_units[i], hidden_units[i+1]) for i in range(len(hidden_units)-1)]

    def forward(self, x):
        for fc in self.fc_layers:
            x = F.relu(fc(x))

        return x

class HMCN(nn.Module):
    def __init__(self, vocab_size, classes, num_kernels, kernel_size, stride=1, emb_size=128,
                 dropout=0.0, padding_index=0):
        """
        多标签多分类 层次分类
        :param vocab_size: 词表大小
        :param classes: list 层级分类 [33, 100] 每一级对应的类别数
        :param num_kernels: 类别数
        :param kernel_size: 卷积核数量(channels数)
        :param stride: stride
        :param emb_size: 词向量维度
        :param dropout: dropout值
        :param padding_index:
        """
        super(HMCN, self).__init__()
        self.classes = classes
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_index)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_kernels, (k, emb_size), stride) for k in kernel_size])
        self.dropout = nn.Dropout(dropout)
        self.fc_layers = [nn.Linear(num_kernels * len(kernel_size), c)
                          if i == 0 else nn.Linear(num_kernels * len(kernel_size) + self.classes[i-1], c)
                          for i, c in enumerate(self.classes)]

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, (int(x.size(2)),)).squeeze(2)
        return x

    def forward(self, x):
        # embedding
        emb = self.emb(x)
        emb = emb.unsqueeze(1)
        conv_out = torch.cat([self.conv_and_pool(emb, conv) for conv in self.convs], 1)
        conv_out = self.dropout(conv_out)
        out = {}
        for i, c in enumerate(self.classes):
            if i == 0:
                # 一级不需要其他特征
                fc_out = self.fc_layers[i](conv_out)
                out[i] = fc_out
            else:
                # 其他级需要上级的特征
                conv_super_out = torch.cat([out[i-1], conv_out], 1)
                fc_out = self.fc_layers[i](conv_super_out)
                out[i] = fc_out

        return out
