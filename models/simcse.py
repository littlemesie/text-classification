# -*- coding:utf-8 -*-

"""
@date: 2023/8/24 上午11:03
@summary: 基于Simcse 做分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel, BertConfig

class Similarity(nn.Module):
    """计算相似性"""
    def __init__(self, norm=True):
        super().__init__()
        self.norm = norm

    def forward(self, x, y, temp=1.0):
        x, y = x.float(), y.float()  # for float16 optimization, need to convert to float
        if self.norm:
            x = x / torch.norm(x, dim=-1, keepdim=True)
            y = y / torch.norm(y, dim=-1, keepdim=True)
        return torch.matmul(x, y.t()) / temp

class SimCSE(nn.Module):
    def __init__(self, model_path, device, embed_size=128, pooling="cls", scale=0.05, dropout=0.0):
        super(SimCSE, self).__init__()
        self.device = device
        self.pooling = pooling
        self.sacle = scale
        self.config = BertConfig.from_pretrained(model_path)
        self.ptm = AlbertModel.from_pretrained(model_path, self.config)
        self.sim = Similarity(norm=True)
        self.dropout = nn.Dropout(dropout)
        self.dnn = nn.Linear(self.config.hidden_size, embed_size)

    def get_pooled_embedding(self, input_ids, token_type_ids, attention_mask):
        """"""
        out = self.ptm(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

    def cosine_sim(self, embed1, embed2):
        """"""
        cosine_sim = F.cosine_similarity(embed1, embed2, dim=-1)
        return cosine_sim

    def forward(self, input_ids, token_type_ids, attention_mask):
        """"""
        out = self.get_pooled_embedding(input_ids, token_type_ids, attention_mask)
        # out = self.dropout(out)
        out = self.dnn(out)
        return out

    def supervised_loss_(self, cosine_sim, target):
        """
        有监督loss计算
        :param cosine_sim: tensor([0.3132,  0.7630])
        :param target:tensor([0, 1])
        :return:
        """
        # cosine_sim = cosine_sim / self.sacle
        cosine_sim = cosine_sim.sigmoid().unsqueeze(1)
        target = target.unsqueeze(1).float()
        loss = F.binary_cross_entropy(input=cosine_sim, target=target, reduction='mean')
        return loss

    def supervised_loss(self, cosine_sim, target):
        """
        有监督loss计算
        :param cosine_sim:
        :param target:
        :return:
        """
        # cosine_sim = cosine_sim / self.sacle
        cosine_sim = F.sigmoid(cosine_sim)
        loss = F.cross_entropy(input=cosine_sim, target=target)
        return loss

    def contrastive_loss(self, embed1, embed2, target):
        """对比学习loss计算"""
        cosine_sim = self.sim(embed1, embed2)
        p_loss = self.supervised_loss(cosine_sim, target[0])
        scl_loss = self.supervised_loss(cosine_sim, target[1])
        return p_loss, scl_loss
