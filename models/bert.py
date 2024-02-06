# -*- coding:utf-8 -*-

"""
@date: 2024/2/6 上午11:37
@summary:
"""
import torch
from torch import nn
import torch.nn.functional as F

class BertBase(nn.Module):
    def __init__(self, pretrained_model, ptm_config, pooling='cls', dropout=0.1, output_emb_size=256):
        super(BertBase, self).__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        self.lin = nn.Linear(ptm_config.hidden_size, output_emb_size)
        self.output_emb_size = output_emb_size

    def get_pooled_embedding(self, input_ids, token_type_ids, attention_mask):
        """"""
        out = self.ptm(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        if self.pooling == 'cls':
            cls_embedding = out.last_hidden_state[:, 0]  # [batch, 768]

        elif self.pooling == 'pooler':
            cls_embedding = out.pooler_output  # [batch, 768]

        elif self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            cls_embedding = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        elif self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            cls_embedding = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        else:
            raise "pooling is error!"

        if self.output_emb_size > 0:
            cls_embedding = self.lin(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, dim=-1)
        return cls_embedding

    def forward(self, *args):
        pass

class BertClassfication(BertBase):
    """"""
    def __init__(self, pretrained_model, ptm_config, device, num_classes, pooling='cls', dropout=None, emb_size=256):
        super(BertClassfication, self).__init__(pretrained_model, ptm_config, pooling, dropout, emb_size)
        self.device = device
        self.num_classes = num_classes
        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, input_ids, token_type_ids, attention_mask):
        embed = self.get_pooled_embedding(input_ids, token_type_ids, attention_mask)
        out = self.fc(embed)
        return out