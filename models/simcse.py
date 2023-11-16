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

class UTCLoss:
    """"""
    def __call__(self, logit, label):
        return self.forward(logit, label)

    def forward(self, logit, label):
        logit = (1.0 - 2.0 * label) * logit
        logit_neg = logit - label * 1e12
        logit_pos = logit - (1.0 - label) * 1e12
        zeros = torch.zeros_like(logit[..., :1])
        logit_neg = torch.concat([logit_neg, zeros], dim=-1)
        logit_pos = torch.concat([logit_pos, zeros], dim=-1)
        label = torch.concat([label, zeros], dim=-1)
        logit_neg[label == -100] = -1e12
        logit_pos[label == -100] = -1e12
        neg_loss = torch.logsumexp(logit_neg, dim=-1)
        pos_loss = torch.logsumexp(logit_pos, dim=-1)
        loss = (neg_loss + pos_loss).mean()
        return loss

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

    def _rgl_loss(self, embed1, category, labels, equal_type="raw", alpha_rgl=0.5):
        """
        Compute the label consistency loss of sentence embeddings per batch.
        Please refer to https://aclanthology.org/2022.findings-naacl.81/
        for more details.
        """
        (category_token, category_segment, category_mask) = category

        batch_size = embed1.shape[0]
        loss = 0
        scores_list = []
        for i in range(batch_size):
            score_list = []
            for j in range(category_token.shape[0]):
            # for j in range(2):
                category_emb = self.forward(category_token[j].unsqueeze(0), category_segment[j].unsqueeze(0),
                                            category_mask[j].unsqueeze(0))
                # print(category_emb)
                score = F.cosine_similarity(embed1[i].unsqueeze(0), category_emb, dim=-1)
                score_list.append(score.detach().numpy()[0])
            # print(score_list)
            scores_list.append(score_list)
        logits = torch.Tensor(scores_list)
        loss += F.cross_entropy(logits, labels)

        return loss

    def forward(self, input_ids, token_type_ids, attention_mask):
        """"""
        out = self.get_pooled_embedding(input_ids, token_type_ids, attention_mask)
        # out = self.dropout(out)
        out = self.dnn(out)
        return out


class SimCSECNN(nn.Module):
    def __init__(self, vocab_size, num_kernels, kernel_size, stride=1, emb_size=128, hidden_size=128,
                 dropout=0.5, padding_index=0):
        """
        :param vocab_size: 词表大小
        :param num_kernels: 卷积核数量(channels数)
        :param kernel_size:  卷积核尺寸
        :param stride: stride
        :param emb_size: 词向量维度
        :param dropout: dropout值
        :param padding_index: padding_index
        """
        super(SimCSECNN, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_index)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_kernels, (k, emb_size), stride) for k in kernel_size])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_kernels * len(kernel_size), hidden_size)
        self.sim = Similarity(norm=True)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def get_pooled_embedding(self, x):
        emb = self.emb(x)
        emb = emb.unsqueeze(1)
        emb = torch.cat([self.conv_and_pool(emb, conv) for conv in self.convs], 1)
        emb = self.dropout(emb)
        emb = self.fc(emb)

        return emb

    def forward(self, x):
        out = self.get_pooled_embedding(x)
        return out

    def _rgl_loss(self, text_embed, category_embed, labels, device, equal_type="raw", alpha_rgl=0.5):
        """
        Compute the label consistency loss of sentence embeddings per batch.
        Please refer to https://aclanthology.org/2022.findings-naacl.81/
        for more details.
        """
        loss = 0
        scores_list = []
        batch_size = text_embed.shape[0]
        for i in range(batch_size):
            score_list = []
            for j in range(category_embed.shape[0]):
                # print(text_embed[i].unsqueeze(0), category_embed[j])
                score = self.sim(text_embed[i].unsqueeze(0), category_embed[j])
                # print(score)
                # break
                # score = F.cosine_similarity(text_embed[i].unsqueeze(0), category_embed[j], dim=-1)
                score_list.append(score.detach().cpu().numpy()[0])
            scores_list.append(score_list)
        logits = torch.Tensor(scores_list).to(device)
        # logits = logits.softmax(dim=1)
        logits = logits.sigmoid()
        loss += F.cross_entropy(logits, labels)
        # loss += F.binary_cross_entropy(logits, labels)
        return loss