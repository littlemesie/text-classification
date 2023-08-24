# -*- coding:utf-8 -*-

"""
@date: 2023/3/29 下午6:51
@summary:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextRCNN(nn.Module):
    def __init__(self,  vocab_size, num_classes, hidden_size, num_layers=1, emb_size=128, dropout=0.0,
                 pad_size=300, embed_pretrained=None):
        """
        :param vocab_size: 词表大小
        :param num_classes: 类别数
        :param hidden_size: lstm隐藏层
        :param num_layers: lstm层数
        :param emb_size: 词向量维度
        :param dropout: dropout值
        :param pad_size: 每句话处理成的长度(短填长切)
        :param embed_pretrained: 词向量文件
        """
        super(TextRCNN, self).__init__()
        if embed_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embed_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size - 1)

        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, bidirectional=True, batch_first=True,
                            dropout=dropout)
        self.maxpool = nn.MaxPool1d(pad_size)
        self.fc = nn.Linear(hidden_size * 2 + emb_size, num_classes)  # 将embedding层与LSTM输出拼接，并进行非线性激活

    def forward(self, x):

        # 模型输入： [batch_size, seq_len]
        embed = self.embedding(x)  # 经过embedding层[batch_size, seq_len, embed_size]=[64, 32, 64]

        out, _ = self.lstm(embed)  # 双向LSTM：隐层大小为hidden_size，得到所有时刻的隐层状态(前向隐层和后向隐层拼接)
        # [batch_size, seq_len, hidden_size * 2]
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
