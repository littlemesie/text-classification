# -*- coding:utf-8 -*-

"""
@date: 2024/2/6 下午2:55
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
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoConfig, AutoModel
from sklearn.metrics import f1_score
from core.model_config import get_model_config
from models.bert import BertClassfication
from utils.time_util import get_time_diff

config = get_model_config('bert')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BertDataset(Dataset):
    def __init__(self, filepath):
        """
        Args:
            filepath: 文件路径
        """
        super(BertDataset, self).__init__()
        self.texts, self.labels = self.load_data(filepath)

    def load_data(self, path):
        data = pd.read_csv(path, error_bad_lines=False)
        data = data.dropna()
        texts = data.text.to_list()
        labels = data.label.to_list()
        return texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        return text, label

class BatchBertDataset:
    """
    call function for tokenizing and getting batch text
    """
    def __init__(self, tokenizer, class_num, label2id, max_len=312, mode='val'):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        self.class_num = class_num
        self.label2id = label2id

    def text2id(self, batch_text):
        return self.tokenizer(batch_text, max_length=self.max_len,
                              truncation=True, padding=True, return_tensors='pt')

    def __call__(self, batch):
        batch_text = [item[0] for item in batch]
        batch_label = []
        for item in batch:
            label_list = [self.label2id[i] for i in item[1].split(',')]
            l_list = [float(1) if i in label_list else float(0) for i in range(self.class_num)]
            batch_label.append(l_list)

        text_source = self.text2id(batch_text)
        batch_data = {
            'input_ids': text_source.get('input_ids').squeeze(1),
            'attention_mask': text_source.get('attention_mask').squeeze(1),
            'token_type_ids': text_source.get('token_type_ids').squeeze(1),
            'batch_label': torch.tensor(batch_label, dtype=torch.float32)

        }
        return batch_data

def load_tokenizer(model_path, special_token=None):
    """load tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if special_token:
        tokenizer.add_special_tokens(special_token)
    return tokenizer

def get_train_dataloader(tokenizer, class_num, label2id):
    """load dataset"""
    dataset_batch = BatchBertDataset(tokenizer, class_num, label2id, max_len=config.max_seq_length)
    train_dataset = BertDataset(f"{config.data_path}/train_1.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)

    dev_dataset = BertDataset(f"{config.data_path}/dev.csv")
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)

    return train_dataloader, dev_dataloader



def get_label_data():
    """label data"""
    label_df = pd.read_csv(f"{config.data_path}/label.csv")
    label_df = label_df.dropna()
    id2label = {}
    label2id = {}
    for ind, row in label_df.iterrows():
        id2label[ind] = row['label']
        label2id[row['label']] = ind
    return id2label, label2id


def evaluation(model, valid_dataloader, criterion):
    """evaluation"""
    model.eval()

    y_label = []
    pred_label = []
    for ind, batch_data in enumerate(valid_dataloader):
        input_ids = batch_data['input_ids'].to(device)
        token_type_ids = batch_data['token_type_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        labels = batch_data['batch_label']
        y_label = y_label + labels.tolist()
        out = model(input_ids, token_type_ids, attention_mask)
        # loss = criterion(out, labels)
        probs = F.sigmoid(out)
        for p in probs:
            p_label = [1 if i > 0.5 else 0 for i in p]
            pred_label.append(p_label)


    micro_f1_score = f1_score(y_pred=pred_label, y_true=y_label, average="micro")

    return micro_f1_score


def train(load_model=False):
    """train model"""
    tokenizer = load_tokenizer(config.model_path)

    id2label, label2id = get_label_data()
    train_dataloader, dev_dataloader = get_train_dataloader(tokenizer, len(label2id), label2id)
    ptm_config = AutoConfig.from_pretrained(config.model_path)
    ptm = AutoModel.from_pretrained(config.model_path)
    model = BertClassfication(
        ptm,
        ptm_config,
        device,
        num_classes=len(id2label),
        pooling=config.pooling,
        dropout=config.dropout,
        emb_size=config.output_emb_size
    )
    # model path
    best_model_path = f"{config.best_model}.pt"
    if load_model:
        model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    model.to(device)
    num_train_optimization_steps = len(train_dataloader) * config.epochs
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=float(config.learning_rate), correct_bias=not config.bertadam)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_optimization_steps * config.warmup_proportion),
                                                             num_train_optimization_steps)

    model.train()

    total_batch = 0
    best_loss = 0.00001
    criterion = torch.nn.BCEWithLogitsLoss()
    start_time = time.time()
    for epoch in range(config.epochs):
        tqdm_bar = tqdm(train_dataloader, desc="training epoch{epoch}".format(epoch=epoch))
        loss_total, loss_diffs = [], []
        epoch_time = time.time()
        early_stop_batch = 0
        for batch_idx, batch_data in enumerate(tqdm_bar):
            input_ids = batch_data['input_ids'].to(device)
            token_type_ids = batch_data['token_type_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            labels = batch_data['batch_label'].to(device)
            # loss
            out = model(input_ids, token_type_ids, attention_mask)
            # print(out.shape, labels.shape)
            loss = criterion(input=out, target=labels)

            # print(loss)
            # micro_f1_score = evaluation(model, dev_dataloader, criterion)
            # print(micro_f1_score)

            loss.backward()
            optimizer.zero_grad()
            scheduler.step()
            optimizer.step()
            loss_total.append(loss.detach().item())
            if batch_idx != 0:
                loss_diffs.append(abs(loss.detach().item() - loss_total[-2]))
            if total_batch % 1000 == 0 and total_batch != 0:
                micro_f1_score = evaluation(model, dev_dataloader, criterion)
                # 保存模型
                mean_diff_loss = np.mean(loss_diffs[-10:])
                if best_loss > mean_diff_loss:

                    best_loss = mean_diff_loss
                    # 保存最好好的模型
                    torch.save(model.state_dict(), best_model_path)
                    early_stop_batch += 1
                # 提前终止训练
                if early_stop_batch == 10:
                    torch.save(model.state_dict(), best_model_path)
                    return

                time_diff = get_time_diff(epoch_time)
                msg = "Iter：{0:6}, Train_Loss: {1:>5.2}, Micro_f1_score: {2:5.2}, Time: {3}"
                print(msg.format(total_batch, loss.item(), micro_f1_score, time_diff))

            total_batch += 1
            print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))
            # break
            # 最后一轮保存模型
        if epoch == config.epochs - 1:
            torch.save(model.state_dict(), best_model_path)

def predict(text, tokenizer):
    """predict"""
    # load label
    id2label, label2id = get_label_data()
    # load model
    ptm_config = AutoConfig.from_pretrained(config.model_path)
    ptm = AutoModel.from_pretrained(config.model_path)
    model = BertClassfication(
        ptm,
        ptm_config,
        device,
        num_classes=len(id2label),
        pooling=config.pooling,
        dropout=config.dropout,
        emb_size=config.output_emb_size
    )
    # model path
    best_model_path = f"{config.best_model}.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    model.eval()

    t1 = time.time()
    label_source = tokenizer([text], max_length=config.max_seq_length,
                             truncation=True, padding=True, return_tensors='pt')
    input_ids = label_source.get('input_ids').squeeze(1).to(device)
    attention_mask = label_source.get('attention_mask').squeeze(1).to(device)
    token_type_ids = label_source.get('token_type_ids').squeeze(1).to(device)
    # print(input_ids)
    out = model(input_ids, token_type_ids, attention_mask)

    probs = torch.sigmoid(out)
    print(probs)
    labels = []
    for prob in probs.tolist():
        for i, p in enumerate(prob):
            if p > 0.1:
                label = id2label[i]
                labels.append((label, round(p, 4)))
    print(labels)
    print(time.time() - t1)

if __name__ == '__main__':
    """
    数据集：链接: https://pan.baidu.com/s/1n5RsF5y-1HUGbm6GCv76hg?pwd=rdxx 提取码: rdxx
    """
    train(load_model=False)
    # tokenizer = load_tokenizer(config.model_path)
    # text = '十个月婴儿的能力.刚满十个月婴儿应具备哪些能力?'
    # print(text)
    # predict(text, tokenizer)

