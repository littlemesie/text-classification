# -*- coding:utf-8 -*-

"""
@date: 2024/1/30 下午6:02
@summary: 基于检索进行分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticIndexBase(nn.Module):
    def __init__(self, pretrained_model, ptm_config, pooling='cls', dropout=0.1, output_emb_size=256):
        super(SemanticIndexBase, self).__init__()
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

    def get_semantic_embedding(self, data_loader):
        self.eval()
        with torch.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids, attention_mask = batch_data
                text_embeddings = self.get_pooled_embedding(input_ids, token_type_ids, attention_mask)
                yield text_embeddings

    def cosine_sim(
        self,
        query_input_ids,
        title_input_ids,
        query_token_type_ids=None,
        query_attention_mask=None,
        title_token_type_ids=None,
        title_attention_mask=None,
    ):

        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_attention_mask
        )

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_attention_mask
        )

        cosine_sim = torch.sum(query_cls_embedding * title_cls_embedding, dim=-1)
        return cosine_sim

    def forward(self):
        pass

class SemanticIndexBatchNeg(SemanticIndexBase):
    def __init__(self, pretrained_model, ptm_config, device, pooling='cls', dropout=None, margin=0.3, scale=30, output_emb_size=256):
        super(SemanticIndexBatchNeg, self).__init__(pretrained_model, ptm_config, pooling, dropout, output_emb_size)
        self.margin = margin
        # Used scaling cosine similarity to ease converge
        self.scale = scale
        self.device = device

    def forward(
        self,
        query_input_ids,
        title_input_ids,
        query_token_type_ids=None,
        query_attention_mask=None,
        title_token_type_ids=None,
        title_attention_mask=None
    ):

        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_attention_mask
        )

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_attention_mask
        )

        cosine_sim = torch.matmul(query_cls_embedding, title_cls_embedding.t())

        # Subtract margin from all positive samples cosine_sim()
        margin_diag = torch.full([query_cls_embedding.shape[0]],
                                 fill_value=self.margin, dtype=torch.get_default_dtype()).to(self.device)

        cosine_sim = cosine_sim - torch.diag(margin_diag)

        # Scale cosine to ease training converge
        cosine_sim *= self.scale

        labels = torch.arange(0, query_cls_embedding.shape[0]).to(self.device)

        # labels = torch.reshape(labels, shape=[-1, 1])

        loss = F.cross_entropy(input=cosine_sim, target=labels)

        return loss