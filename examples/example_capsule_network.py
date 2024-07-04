# -*- coding:utf-8 -*-

"""
@date: 2024/6/6 上午9:49
@summary:
"""
import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from models.capsule_network import CapsuleNetwork
from core.model_config import get_model_config
from utils.time_util import get_time_diff

config = get_model_config('capsule_network')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

label_ind = {
    "negative": 0, "矛盾纠纷": 1, "人际纠纷": 2, "治安纠纷": 3, "打架纠纷": 4, "草场纠纷": 5, "出租车纠纷": 6, "中介纠纷": 7,
    "医托纠纷": 8, "假币纠纷": 9, "家暴纠纷": 10, "教育问题纠纷": 11, "退役军人问题纠纷": 12, "家庭婚姻感情纠纷": 13,
    "经济纠纷": 14, "邻里纠纷": 15, "劳资纠纷": 16, "医疗纠纷": 17
}

def load_tokenizer(model_path, special_token=None):
    """load tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if special_token:
        tokenizer.add_special_tokens(special_token)
    return tokenizer

class TextDataset(Dataset):
    def __init__(self, filepath):
        super(TextDataset, self).__init__()
        self.texts, self.labels = self.load_data(filepath)

    def load_data(self, path):
        train = pd.read_csv(path)
        train = train[['text', 'label']]
        train['label'] = train['label'].apply(lambda x: eval(x))
        texts = train.text.to_list()
        labels = train.label.to_list()
        return texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        return text, label

class BatchTextDataset(object):
    def __init__(self, tokenizer, label_ind=None, max_len=512, padding=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_ind = label_ind
        self.padding = padding

    def text2id(self, batch_text):
        return self.tokenizer(batch_text, max_length=self.max_len,
                              truncation=True, padding=self.padding, return_tensors='pt')

    def __call__(self, batch):
        batch_text = [item[0] for item in batch]
        batch_label = [item[1] for item in batch]
        text_source = self.text2id(batch_text)
        batch_tensor_token = text_source.get('input_ids').squeeze(1)
        if self.label_ind:
            batch_label_ = []
            for label in batch_label:
                label_ = [0 for i in self.label_ind.keys()]
                for l in label:
                    label_[self.label_ind[l]] = 1
                batch_label_.append(label_)
            batch_tensor_label = torch.tensor(batch_label_)
        else:
            batch_tensor_label = torch.tensor(batch_label)

        return batch_tensor_token, batch_tensor_label

def get_train_dataloader(tokenizer, data_path, al_ind):
    """获取数据"""

    dataset_batch = BatchTextDataset(tokenizer, label_ind=al_ind, max_len=config.max_length, padding='max_length')

    train_dataset = TextDataset(f"{data_path}/train.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)

    dev_dataset = TextDataset(f"/{data_path}/dev.csv")
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)

    test_dataset = TextDataset(f"/{data_path}/test.csv")
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)

    return train_dataloader, dev_dataloader, test_dataloader

def calc_accuracy(batch_ture, batch_pred):
    """"""
    acc_list = []
    for i, bt in enumerate(batch_ture):
        index_list = [j for j, b in enumerate(bt) if b != 0 and batch_ture[i][j] != 0]
        y_true = [bt[ind] for ind in index_list]
        y_pred = [batch_pred[i][ind] for ind in index_list]
        acc = metrics.accuracy_score(y_true, y_pred)
        acc_list.append(acc)
    acc_avg = sum(acc_list) / len(acc_list)
    return acc_avg

def evaluation(model, test_dataloader, loss_func, device):
    """模型评估"""
    model.eval()
    total_loss = 0
    predict_all = []
    labels_all = []
    for ind, (token, label) in enumerate(test_dataloader):
        token = token.to(device)
        label = label.to(device)
        _, out = model(token)
        out = out.squeeze(2)
        label = label.float()
        loss = loss_func(out, label)
        total_loss += loss.detach().item()
        label = label.data.cpu().numpy()
        pred_label = out.data.cpu().sigmoid().numpy()
        for l in label:
            labels_all.append(l.tolist())
        for pl in pred_label:
            pl_ = [1 if p > config.sigmoid_threshold else 0 for p in pl.tolist()]
            predict_all.append(pl_)
    # acc = metrics.accuracy_score(labels_all, predict_all)
    acc = calc_accuracy(labels_all, predict_all)
    return acc, total_loss / len(test_dataloader)

def train(al_ind, data_path, tokenizer, load_model=False):
    start_time = time.time()
    train_dataloader, valid_dataloader, test_dataloader = get_train_dataloader(tokenizer, data_path, al_ind)
    num_class = len(al_ind)
    model = CapsuleNetwork(
        config.vocab_size,
        num_class,
        config.kernel_size,
        num_kernels=config.num_kernels,
        stride=config.stride,
        emb_size=config.emb_size,
        max_length=config.max_length,
        dim_capsule=config.dim_capsule,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_compressed_capsule=config.num_compressed_capsule,
        dropout=config.dropout,
        route=config.route,
        fc=config.fc,
        device=device
    )


    best_model_path = f"capsule_network_best.pt"
    if load_model:
        model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(config.learning_rate),
                                  betas=(config.betas[0], config.betas[1]),
                                  eps=float(config.eps),
                                  weight_decay=config.weight_decay,
                                  amsgrad=config.amsgrad)
    # loss_func = F.cross_entropy
    loss_func = F.binary_cross_entropy_with_logits
    # loss_func = F.binary_cross_entropy
    loss_total, top_acc = [], 0
    valid_best_loss = float('inf')
    last_improve = 0
    total_batch = 0
    stop_flag = False
    train_acc_history, train_loss_history = [], []
    valid_acc_history, valid_loss_history = [], []

    for epoch in range(config.epochs):
        print("Epoch [{}/{}]".format(epoch + 1, config.epochs))
        model.train()
        epoch_time = time.time()
        tqdm_bar = tqdm(train_dataloader, desc="training epoch{epoch}".format(epoch=epoch))
        for i, (token, label) in enumerate(tqdm_bar):
            # data to device
            token = token.to(device)
            label = label.to(device)
            # print(token.shape)
            model.zero_grad()
            _, out = model(token)
            out = out.squeeze(2)
            label = label.float()
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()
            loss_total.append(loss.detach().item())
            if total_batch % config.validate_term_num == 0 and total_batch != 0:
                valid_acc, valid_loss = evaluation(model, valid_dataloader, loss_func, device)
                true_label = label.data.cpu().numpy()
                # pred_label = out.data.numpy()
                pred_label = out.data.cpu().sigmoid().numpy()
                true_label_, pred_label_ = [], []
                for tl in true_label:
                    true_label_.append(tl.tolist())
                for pl in pred_label:
                    pl_ = [1 if p > config.sigmoid_threshold else 0 for p in pl.tolist()]
                    pred_label_.append(pl_)

                # train_acc = metrics.accuracy_score(true_label_, pred_label_)
                train_acc = calc_accuracy(true_label_, pred_label_)

                train_acc_history.append(train_acc)
                train_loss_history.append(loss.detach().item())
                valid_acc_history.append(valid_acc)
                valid_loss_history.append(valid_loss)

                if epoch and epoch % 5 == 0:
                    save_model_path = f"capsule_network_{epoch}.pt"
                    torch.save(model.state_dict(), save_model_path)

                # evaluate on validate data
                if valid_loss < valid_best_loss:
                    valid_best_loss = valid_loss
                    # 保存最好好的模型
                    torch.save(model.state_dict(), best_model_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ""
                time_diff = get_time_diff(epoch_time)
                msg = "Iter：{0:6}, Train_Loss: {1:>5.2}, Train_Acc: {2:>6.2%}, Val_Loss: {3:5.2}, Val_Acc: {4:6.2%}, Time: {5} {6}"
                print(msg.format(total_batch, loss.item(), train_acc, valid_loss, valid_acc, time_diff, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > 2000:
                print('No optimization for a long time, auto-stopping......')
                stop_flag = True
                break
        # 最后一轮保存模型
        if epoch == config.epochs - 1:
            torch.save(model.state_dict(), best_model_path)
        print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))
        if stop_flag:
            break
        time.sleep(0.5)

def predict(sent, label_ind, tokenizer):
    num_class = len(label_ind)
    model = CapsuleNetwork(
        config.vocab_size,
        num_class,
        config.kernel_size,
        num_kernels=config.num_kernels,
        stride=config.stride,
        emb_size=config.emb_size,
        max_length=config.max_length,
        dim_capsule=config.dim_capsule,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_compressed_capsule=config.num_compressed_capsule,
        dropout=config.dropout,
        route=config.route,
        fc=config.fc,
        device=device
    ).to(device)
    best_model_path = f"capsule_network_best.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        encodings_dict = tokenizer(sent, truncation=True, max_length=config.max_length, padding='max_length')
        input_ids = encodings_dict['input_ids']
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        _, out = model(input_ids)
        out = out.squeeze(2)
        print(out)
        pred_label = out.data.cpu().sigmoid().numpy()
    return pred_label[0]

if __name__ == '__main__':
    """"""
    # data_path = "/home/mesie/python/aia-nlp-service/data/alarm_label/subclass2_small/纠纷/train"
    # data_path = "/home/nlp/python/nlp/aia-nlp-service/data/alarm_label/subclass2_small/纠纷/train"
    tokenizer = load_tokenizer(config.model_path)
    # train(label_ind, data_path, tokenizer, load_model=False)
    text = "打架引发纠纷，现其中一方被咬伤"
    pred_label_prob = predict(text, label_ind, tokenizer)
    pred_label_prob = [(round(p, 4)) for p in pred_label_prob]  # 取小数点位数
    multi_ind_labels = {v: k for k, v in label_ind.items()}
    print(pred_label_prob)
    pred_res = [(multi_ind_labels[i], p) for i, p in enumerate(pred_label_prob) if p > 0.9]
    print(pred_res)