# -*- coding:utf-8 -*-

"""
@date: 2023/8/24 下午5:45
@summary: 使用simcse分类
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
from models.simcse import SimCSE
from utils.dataloader import SimCSEDataset, BatchSimCSEDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = "/media/mesie/F0E66F06E66ECC82/数据/category/simcse"
# data_path = "/home/yons/python/nlp/data/jj"

def load_tokenizer(model_path, special_token=None):
    """load tokenizer"""
    tokenizer = BertTokenizer.from_pretrained(model_path)
    if special_token:
        tokenizer.add_special_tokens(special_token)
    return tokenizer

def get_train_dataloader(tokenizer):
    """load dataset"""
    dataset_batch = BatchSimCSEDataset(tokenizer, max_len=256)
    train_dataset = SimCSEDataset(f"{data_path}/jj_train.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)
    dev_dataset = SimCSEDataset(f"{data_path}/jj_dev.csv")
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)

    return train_dataloader, dev_dataloader

def evaluation(model, valid_dataloader):
    """model evaluation"""
    model.eval()
    total_loss = 0
    pred_list = np.array([], dtype=int)
    labels = np.array([], dtype=int)

    for ind, batch_data in enumerate(valid_dataloader):
        text_token = batch_data['text_token'].to(device)
        text_segment = batch_data['text_segment'].to(device)
        text_mask = batch_data['text_mask'].to(device)
        category_token = batch_data['category_token'].to(device)
        category_segment = batch_data['category_segment'].to(device)
        category_mask = batch_data['category_mask'].to(device)
        label = batch_data['label'].to(device)
        text_emb = model(text_token, text_segment, text_mask)
        category_emb = model(category_token, category_segment, category_mask)
        cosine_sim = model.sim(text_emb, category_emb)
        # cosine_sim = model.cosine_sim(text_emb, category_emb)
        loss = model.supervised_loss(cosine_sim, label)
        # loss = model.supervised_loss_(cosine_sim, label)
        total_loss += loss.detach().item()

        label = label.data.cpu().numpy()
        pred = cosine_sim.mean(1).data.cpu().sigmoid().numpy()
        # pred = cosine_sim.data.cpu().sigmoid().numpy()
        # 根据阈值进行判断
        pred = [1 if p > 0.5 else 0 for p in pred]
        labels = np.append(labels, label)
        pred_list = np.append(pred_list, pred)
    # calc acc
    acc = metrics.accuracy_score(labels, pred_list)
    return acc, total_loss / len(valid_dataloader)

def train(tokenizer, model_path):
    """train model"""
    train_dataloader, dev_dataloader = get_train_dataloader(tokenizer)
    model = SimCSE(model_path, device, embed_size=128)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model.train()
    best_model_path = "simcse_classification_best.pt"

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
            text_segment = batch_data['text_segment'].to(device)
            text_mask = batch_data['text_mask'].to(device)
            category_token = batch_data['category_token'].to(device)
            category_segment = batch_data['category_segment'].to(device)
            category_mask = batch_data['category_mask'].to(device)
            label = batch_data['label'].to(device)
            text_emb = model(text_token, text_segment, text_mask)
            category_emb = model(category_token, category_segment, category_mask)
            cosine_sim = model.sim(text_emb, category_emb)
            # cosine_sim = model.cosine_sim(text_emb, category_emb)
            # print(cosine_sim)
            # print(cosine_sim.mean(1).data.cpu())
            # loss
            loss = model.supervised_loss(cosine_sim, label)

            # loss = model.supervised_loss_(cosine_sim, label)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total.append(loss.detach().item())
            if batch_idx != 0:
                loss_diffs.append(abs(loss.detach().item() - loss_total[-2]))
            if total_batch % 1000 == 0 and total_batch != 0:
                valid_acc, valid_loss = evaluation(model, dev_dataloader)
                true_label = label.data.cpu()
                pred = cosine_sim.mean(1).data.cpu().sigmoid().numpy()
                # pred = cosine_sim.data.cpu().sigmoid().numpy()
                # 根据阈值进行判断
                pred_list = [1 if p > 0.5 else 0 for p in pred]
                train_acc = metrics.accuracy_score(true_label, pred_list)
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
                msg = "Iter：{0:6}, Train_Loss: {1:>5.2}, Train_Acc: {2:>6.2%}, Val_Loss: {3:5.2}, Val_Acc: {4:6.2%}, Time: {5}"
                print(msg.format(total_batch, loss.item(), train_acc, valid_loss, valid_acc, time_diff))

            total_batch += 1
            print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))
            # break
        # 最后一轮保存模型
        if epoch == epochs - 1:
            torch.save(model.state_dict(), best_model_path)

def generate_embed(model_path, tokenizer):
    """生成embed"""
    best_model_path = f"simcse_classification_best.pt"
    model = SimCSE(model_path, device, embed_size=128)
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    category_df = pd.read_csv(f"{data_path}/category_all.csv")
    dataset_batch = BatchSimCSEDataset(tokenizer, max_len=256)
    all_embeddings = []
    for _, row in category_df.iterrows():
        category = row['category']
        source = dataset_batch.text2id([category])
        category_token = source.get('input_ids').squeeze(1)
        category_mask = source.get('attention_mask').squeeze(1)
        category_segment = source.get('token_type_ids').squeeze(1)
        category_emb = model(category_token, category_segment, category_mask)
        all_embeddings.append(category_emb[0].tolist())
        # print(category_emb)
        # break
    np.save(f"category_embedding", all_embeddings)
    # print(all_embeddings)


def search_category(model_path, tokenizer, text):
    """"""
    best_model_path = f"simcse_classification_best.pt"
    model = SimCSE(model_path, device, embed_size=128)
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    dataset_batch = BatchSimCSEDataset(tokenizer, max_len=256)
    category_df = pd.read_csv(f"{data_path}/category_all.csv")
    category_dict = {i: c for i, c in enumerate(category_df['category'].tolist())}
    embeddings = np.load("category_embedding.npy")
    print(embeddings.shape)
    index = faiss.IndexFlatIP(128)

    # faiss.normalize_L2(embeddings)
    index.add(embeddings)

    source = dataset_batch.text2id([text])
    text_token = source.get('input_ids').squeeze(1)
    text_mask = source.get('attention_mask').squeeze(1)
    text_segment = source.get('token_type_ids').squeeze(1)
    text_emb = model(text_token, text_segment, text_mask)

    D, I = index.search(np.ascontiguousarray(text_emb.detach().numpy()), 10)
    results = [(category_dict[j], D[0][i]) for i, j in enumerate(I[0])]
    print(results)


if __name__ == '__main__':
    """"""
    tokenizer_path = "/home/mesie/python/aia-nlp-service/lib/pretrained/albert_chinese_base"
    # tokenizer_path = "/home/yons/python/conf/lib/pretrained/albert_chinese_base"
    tokenizer = load_tokenizer(tokenizer_path)
    # train(tokenizer, tokenizer_path)
    # generate_embed(tokenizer_path, tokenizer)
    text = "一女称被前男友骗了2万"
    search_category(tokenizer_path, tokenizer, text)


