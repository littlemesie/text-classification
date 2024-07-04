# -*- coding:utf-8 -*-

"""
@date: 2024/6/4 下午7:03
@summary: 基于胶囊网络的多标签分类
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def squash_v1(x, axis):
    """squash函数"""
    s_squared_norm = (x ** 2).sum(axis, keepdim=True)
    scale = torch.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x

def dynamic_routing(b_ij, u_hat, input_capsule_num):
    num_iterations = 3
    for i in range(num_iterations):
        if True:
            leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
            leaky_logits = torch.cat((leak, b_ij), 2)
            leaky_routing = F.softmax(leaky_logits, dim=2)
            c_ij = leaky_routing[:, :, 1:, :].unsqueeze(4)
        else:
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)
        v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3)
        if i < num_iterations - 1:
            b_ij = b_ij + (torch.cat([v_j] * input_capsule_num, dim=1) * u_hat).sum(3)

    poses = v_j.squeeze(1)
    activations = torch.sqrt((poses ** 2).sum(2))
    return poses, activations


def adaptive_kde_routing(batch_size, b_ij, u_hat):
    last_loss = 0.0
    while True:
        if False:
            leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
            leaky_logits = torch.cat((leak, b_ij), 2)
            leaky_routing = F.softmax(leaky_logits, dim=2)
            c_ij = leaky_routing[:, :, 1:, :].unsqueeze(4)
        else:
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)
        c_ij = c_ij/c_ij.sum(dim=1, keepdim=True)
        v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3)
        dd = 1 - ((squash_v1(u_hat, axis=3)-v_j)**2).sum(3)
        b_ij = b_ij + dd

        c_ij = c_ij.view(batch_size, c_ij.size(1), c_ij.size(2))
        dd = dd.view(batch_size, dd.size(1), dd.size(2))

        kde_loss = torch.mul(c_ij, dd).sum()/batch_size
        kde_loss = np.log(kde_loss.item())

        if abs(kde_loss - last_loss) < 0.05:
            break
        else:
            last_loss = kde_loss
    poses = v_j.squeeze(1)
    activations = torch.sqrt((poses ** 2).sum(2))
    return poses, activations


def kde_routing(b_ij, u_hat):
    num_iterations = 3
    for i in range(num_iterations):
        if False:
            leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
            leaky_logits = torch.cat((leak, b_ij), 2)
            leaky_routing = F.softmax(leaky_logits, dim=2)
            c_ij = leaky_routing[:, :, 1:, :].unsqueeze(4)
        else:
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)

        c_ij = c_ij/c_ij.sum(dim=1, keepdim=True)
        v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3)

        if i < num_iterations - 1:
            dd = 1 - ((squash_v1(u_hat, axis=3)-v_j)**2).sum(3)
            b_ij = b_ij + dd
    poses = v_j.squeeze(1)
    activations = torch.sqrt((poses ** 2).sum(2))
    return poses, activations

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.Conv1d(in_channels, out_channels * num_capsules, kernel_size, stride)

        torch.nn.init.xavier_uniform_(self.capsules.weight)

        self.out_channels = out_channels
        self.num_capsules = num_capsules

    def forward(self, x):
        batch_size = x.size(0)
        u = self.capsules(x).view(batch_size, self.num_capsules, self.out_channels, -1, 1)
        poses = squash_v1(u, axis=1)
        activations = torch.sqrt((poses ** 2).sum(1))
        return poses, activations

class FlattenCaps(nn.Module):
    def __init__(self):
        super(FlattenCaps, self).__init__()

    def forward(self, p, a):
        poses = p.view(p.size(0), p.size(2) * p.size(3) * p.size(4), -1)
        activations = a.view(a.size(0), a.size(1) * a.size(2) * a.size(3), -1)
        return poses, activations

class FCCaps(nn.Module):
    def __init__(self, num_classes, input_capsule_num, in_channels, out_channels, route='akde', device='cpu'):
        super(FCCaps, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_capsule_num = input_capsule_num

        self.w1 = nn.Parameter(torch.FloatTensor(1, input_capsule_num, num_classes, out_channels, in_channels))
        torch.nn.init.xavier_uniform_(self.w1)

        self.route = route
        self.device = device


    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_classes, dim=2).unsqueeze(4)

        w1 = self.w1.repeat(batch_size, 1, 1, 1, 1)
        u_hat = torch.matmul(w1, x)

        b_ij = Variable(torch.zeros(batch_size, self.input_capsule_num, self.num_classes, 1)).to(self.device)

        if self.route == 'akde':
            poses, activations = adaptive_kde_routing(batch_size, b_ij, u_hat)
        elif self.route == 'kde':
            poses, activations = kde_routing(b_ij, u_hat)
        else:
            poses, activations = dynamic_routing(b_ij, u_hat, self.input_capsule_num)
        return poses, activations


class CapsuleNetwork(nn.Module):
    def __init__(self, vocab_size, num_classes, kernel_size, num_kernels=32, stride=1, emb_size=128, max_length=512,
                 dim_capsule=16, in_channels=32, out_channels=32, num_compressed_capsule=128,
                 dropout=0.0, padding_index=0, route='akde', fc='fc', device='cpu'):
        super(CapsuleNetwork, self).__init__()
        self.kernel_size = kernel_size
        self.fc = fc
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_index)
        self.convs = nn.ModuleList([nn.Conv1d(max_length, num_kernels, k, stride=stride) for k in kernel_size])
        for i in range(len(self.convs)):
            torch.nn.init.xavier_uniform_(self.convs[i].weight)

        self.primary_capsules = PrimaryCaps(num_capsules=dim_capsule,
                                            in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=1,
                                            stride=1)

        self.flatten_capsules = FlattenCaps()
        self.w = nn.Parameter(torch.FloatTensor(sum([int((emb_size-1*(k-1)-1)/stride)+1 for k in kernel_size])*num_kernels,
                                                num_compressed_capsule))
        torch.nn.init.xavier_uniform_(self.w)

        self.fc_capsules = FCCaps(num_classes=num_classes,
                                  input_capsule_num=num_compressed_capsule,
                                  in_channels=dim_capsule,
                                  out_channels=dim_capsule,
                                  route=route,
                                  device=device)

        self.linear = nn.Linear(dim_capsule*num_classes, num_classes)
        self.batch_morm = nn.BatchNorm1d(dim_capsule*num_classes)
        self.dropout = nn.Dropout(dropout)


    def compression(self, poses, w):
        poses = torch.matmul(poses.permute(0, 2, 1), w).permute(0, 2, 1)
        activations = torch.sqrt((poses ** 2).sum(2))
        return poses, activations

    def forward(self, x):
        emb = self.emb(x)
        nets = torch.cat([F.relu(conv(emb)) for conv in self.convs], 2)
        # nets = self.dropout(nets)
        poses, activations = self.primary_capsules(nets)
        poses, activations = self.flatten_capsules(poses, activations)
        poses, activations = self.compression(poses, self.w)
        poses, activations = self.fc_capsules(poses)
        if self.fc == 'fc':
            poses = poses.view(emb.shape[0], -1)
            output = self.batch_morm(poses)
            activations = self.linear(output)
            return poses, activations.unsqueeze(2)

        return poses, activations

# if __name__ == '__main__':
#     """"""
#     cn = CapsuleNetwork(vocab_size=21128, num_classes=5, kernel_size=[2, 4, 8], max_length=8, stride=1, fc='fc1')
#     x = torch.tensor([[102, 323, 465, 3456, 103, 0, 0, 0], [102, 323, 465, 3456, 1134, 3456, 125, 103]])
#     _, out = cn(x)
#     print(out.squeeze(2))