# -*- coding:utf-8 -*-

"""
@date: 2023/9/15 下午4:51
@summary:
"""
import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import torch
import time
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from transformers import BertTokenizer
from utils.time_util import get_time_diff
from models.simcse import SimCSECNN
from utils.dataloader import SimCSEDataset, BatchSimCSECNNDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = "/media/mesie/F0E66F06E66ECC82/数据/category/simcse"
# data_path = "/home/yons/python/nlp/data/jj"

def load_tokenizer(model_path, special_token=None):
    """load tokenizer"""
    tokenizer = BertTokenizer.from_pretrained(model_path)
    if special_token:
        tokenizer.add_special_tokens(special_token)
    return tokenizer

def category_dataset(tokenizer, category_list, max_len=16):
    """"""
    batch_token = []
    for text in category_list:
        if len(text) > max_len - 2:
            text = text[:max_len - 2]
        token = tokenizer.tokenize(text)
        token = ['[CLS]'] + token + ['[SEP]']
        token_id = tokenizer.convert_tokens_to_ids(token)

        padding = [0] * (max_len - len(token_id))
        token_id = token_id + padding
        batch_token.append(token_id)

    batch_token = torch.tensor(batch_token)

    return batch_token

def get_train_dataloader(tokenizer, category_list):
    """load dataset"""
    dataset_batch = BatchSimCSECNNDataset(tokenizer, max_len=256, category_list=category_list)
    train_dataset = SimCSEDataset(f"{data_path}/jj_train.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)
    dev_dataset = SimCSEDataset(f"{data_path}/jj_dev.csv")
    dev_dataloader = DataLoader(dev_dataset, batch_size=64, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)

    return train_dataloader, dev_dataloader
def evaluation(model, valid_dataloader, category_emb):
    """model evaluation"""
    model.eval()
    total_loss = 0
    for ind, batch_data in enumerate(valid_dataloader):
        text_token = batch_data['text_token'].to(torch.device('cpu'))
        labels = batch_data['labels'].to(torch.device('cpu'))
        text_emb = model(text_token)
        loss = model._rgl_loss(text_emb, category_emb, labels, torch.device('cpu'))
        total_loss += loss.detach().item()

    return total_loss / len(valid_dataloader)

def train(tokenizer, load_model=False):
    """train model"""
    # load category data
    # category_df = pd.read_csv(f"{data_path}/category_all.csv")
    # category_list = category_df['category'].tolist()
    data_df1 = pd.read_csv(f"{data_path}/jj_train.csv")
    data_df2 = pd.read_csv(f"{data_path}/jj_dev.csv")
    category_list = data_df1['category'].tolist() + data_df2['category'].tolist()
    category_list = [i for cl in category_list for i in eval(cl)]
    category_list = list(set(category_list))
    # print(len(category_list))
    category_token = category_dataset(tokenizer, category_list)
    category_token = category_token.to(device)
    # load data
    train_dataloader, dev_dataloader = get_train_dataloader(tokenizer, category_list)
    vocab_size, num_kernels, kernel_size, emb_size = 21128, 100, [3, 4, 5], 128
    model = SimCSECNN(vocab_size, num_kernels, kernel_size, emb_size=emb_size)
    # model path
    best_model_path = "simcsecnn_classification_best.pt"
    if load_model:
        model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=0.001,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=0.01,
                                  amsgrad=False)
    model.train()

    total_batch = 0
    best_loss = 0.0001
    early_stop_batch = 0

    epochs = 1
    start_time = time.time()
    for epoch in range(epochs):
        tqdm_bar = tqdm(train_dataloader, desc="training epoch{epoch}".format(epoch=epoch))
        loss_total, loss_diffs = [], []
        epoch_time = time.time()
        for batch_idx, batch_data in enumerate(tqdm_bar):
            text_token = batch_data['text_token'].to(device)

            labels = batch_data['labels'].to(device)
            text_emb = model(text_token)
            # 全部加入内存太大
            category_emb = model(category_token)
            # loss
            loss = model._rgl_loss(text_emb, category_emb, labels, device)
            loss.requires_grad_(True)
            # print(loss)
            # break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total.append(loss.detach().item())
            if batch_idx != 0:
                loss_diffs.append(abs(loss.detach().item() - loss_total[-2]))
            if total_batch % 500 == 0 and total_batch != 0:
                valid_loss = evaluation(model, dev_dataloader, category_emb)
                # 保存模型
                mean_diff_loss = np.mean(loss_diffs[-10:])
                if best_loss > mean_diff_loss:
                    early_stop_batch = 0
                    best_loss = mean_diff_loss
                    # 保存最好好的模型
                    torch.save(model.state_dict(), best_model_path)
                # 提前终止训练
                early_stop_batch += 1
                if early_stop_batch == 10:
                    torch.save(model.state_dict(), best_model_path)
                    return

                time_diff = get_time_diff(epoch_time)
                msg = "Iter：{0:6}, Train_Loss: {1:>5.2}, Val_Loss: {2:5.2}, Time: {3}"
                print(msg.format(total_batch, loss.item(), valid_loss, time_diff))

            total_batch += 1
            print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))
            # break
            # 最后一轮保存模型
        if epoch == epochs - 1:
            torch.save(model.state_dict(), best_model_path)

def generate_embed(tokenizer):
    """生成embed"""
    best_model_path = f"simcsecnn_classification_best.pt"
    vocab_size, num_kernels, kernel_size, emb_size = 21128, 100, [3, 4, 5], 128
    model = SimCSECNN(vocab_size, num_kernels, kernel_size, emb_size=emb_size)
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    category_df = pd.read_csv(f"{data_path}/category_all.csv")

    all_embeddings = []
    for _, row in category_df.iterrows():
        category = row['category']
        print(category)
        source = tokenizer(category, truncation=True, max_length=16, padding='max_length')
        input_ids = source['input_ids']
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        category_emb = model(input_ids)
        all_embeddings.append(category_emb[0].tolist())
        # print(category_emb[0].tolist())
        # break
    np.save(f"cnn_category_embedding", all_embeddings)
    # print(all_embeddings)


def search_category(tokenizer, text):
    """"""
    best_model_path = f"simcsecnn_classification_best.pt"
    vocab_size, num_kernels, kernel_size, emb_size = 21128, 100, [3, 4, 5], 128
    model = SimCSECNN(vocab_size, num_kernels, kernel_size, emb_size=emb_size)
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    category_df = pd.read_csv(f"{data_path}/category_all.csv")
    category_dict = {i: c for i, c in enumerate(category_df['category'].tolist())}
    embeddings = np.load("cnn_category_embedding.npy")
    print(embeddings.shape)
    index = faiss.IndexFlatIP(128)

    # faiss.normalize_L2(embeddings)
    index.add(embeddings)
    source = tokenizer(text, truncation=True, max_length=16, padding='max_length')
    input_ids = source['input_ids']
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    text_emb = model(input_ids)

    D, I = index.search(np.ascontiguousarray(text_emb.detach().numpy()), 10)
    results = [(category_dict[j], D[0][i]) for i, j in enumerate(I[0])]
    print(results)


if __name__ == '__main__':
    """"""
    tokenizer_path = "/home/mesie/python/aia-nlp-service/lib/pretrained/albert_chinese_base"
    # tokenizer_path = "/home/yons/python/conf/lib/pretrained/albert_chinese_base"
    tokenizer = load_tokenizer(tokenizer_path)
    # train(tokenizer)
    # generate_embed(tokenizer)
    text = "一女称被前男友骗了2万"
    search_category(tokenizer, text)