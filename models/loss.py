# -*- coding:utf-8 -*-

"""
@date: 2024/12/10 下午1:57
@summary:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultiLabelMarginLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MultiLabelMarginLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 确保输入和目标张量的形状一致
        assert inputs.shape == targets.shape, "Input and target must have the same shape."

        # 将目标中的-1替换为一个非常小的数，以避免在计算时产生影响
        mask = (targets != -1).float()
        targets = targets * mask

        # 计算每个正类与最高得分的负类之间的差值
        pos_scores = inputs * targets
        neg_scores = inputs * (1 - targets)
        max_neg_scores, _ = torch.max(neg_scores, dim=-1, keepdim=True)

        # 计算损失
        loss = torch.clamp(1.0 - pos_scores + max_neg_scores, min=0.0) * mask

        if self.reduction == 'mean':
            return loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction option: {self.reduction}")

class MultilabelCrossEntropy(nn.Module):
    """softmax+交叉熵"""
    def __init__(self, reduction='mean'):
        super(MultilabelCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        """"""
        y_pred = (1 - 2 * targets) * inputs
        y_pred_neg = y_pred - targets * 1e12
        y_pred_pos = y_pred - (1 - targets) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.concatenate([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.concatenate([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        loss = neg_loss + pos_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    """使用"""
    # # 假设有一个4类别的多标签分类任务
    # num_classes = 4
    # batch_size = 4
    # inputs = torch.randn(batch_size, num_classes, requires_grad=True)
    # # 真实标签
    # targets = torch.tensor([[0, 1, 1, 0],
    #                         [1, 0, 0, 0],
    #                         [0, 0, 1, 0],
    #                         [1, 0, 1, 0]
    #                         ], dtype=float)
    #
    # criterion = FocalLoss(alpha=0.25, gamma=2)
    # loss = criterion(inputs, targets)
    # print("Focal Loss:", loss.item())
    # # 反向传播
    # loss.backward()
    # MultilabelMarginLoss使用
    # loss_func = MultiLabelMarginLoss()
    # input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.9, 0.5, 0.3, 0.7]], requires_grad=True)
    # target = torch.tensor([[2, 1, 0, -1], [3, 0, -1, 0]])
    # loss = loss_func(input, target)
    # print("Loss:", loss.item())

    # MultilabelCrossEntropy使用
    # loss_func = MultilabelCrossEntropy()
    # input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.9, 0.5, 0.3, 0.7]], requires_grad=True)
    # target = torch.tensor([[0, 0, 1, 1], [1, 0, 0, 1]])
    # loss = loss_func(input, target)
    # print("Loss:", loss)