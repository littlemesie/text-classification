# -*- coding:utf-8 -*-

"""
@date: 2024/2/1 上午9:25
@summary:
"""
import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import time
import torch
import hnswlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
# from transformers.optimization import AdamW
from transformers import AdamW, get_linear_schedule_with_warmup
# from transformers import BertTokenizer, BertConfig, AlbertModel
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from core.model_config import get_model_config
from models.retrieval_match import SemanticIndexBatchNeg
from utils.time_util import get_time_diff

config = get_model_config('retrieval')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RetrievalDataset(Dataset):
    def __init__(self, filepath):
        """
        Args:
            filepath: 文件路径
        """
        super(RetrievalDataset, self).__init__()
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

class BatchRetrievalDataset:
    """
    call function for tokenizing and getting batch text
    """
    def __init__(self, tokenizer, max_len=312, mode='val'):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

    def text2id(self, batch_text):
        return self.tokenizer(batch_text, max_length=self.max_len,
                              truncation=True, padding=True, return_tensors='pt')

    def __call__(self, batch):
        batch_text = [item[0] for item in batch]
        batch_label = [item[1] for item in batch]
        text_source = self.text2id(batch_text)
        if self.mode == 'train':
            label_source = self.text2id(batch_label)
            batch_data = {
                'text_token': text_source.get('input_ids').squeeze(1),
                'text_mask': text_source.get('attention_mask').squeeze(1),
                'text_type_ids': text_source.get('token_type_ids').squeeze(1),
                'label_token': label_source.get('input_ids').squeeze(1),
                'label_mask': label_source.get('attention_mask').squeeze(1),
                'label_type_ids': label_source.get('token_type_ids').squeeze(1),

            }
        else:
            batch_data = {
                'text_token': text_source.get('input_ids').squeeze(1),
                'text_mask': text_source.get('attention_mask').squeeze(1),
                'text_type_ids': text_source.get('token_type_ids').squeeze(1),
                'batch_label': batch_label
            }

        return batch_data

def load_tokenizer(model_path, special_token=None):
    """load tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if special_token:
        tokenizer.add_special_tokens(special_token)
    return tokenizer

def get_train_dataloader(tokenizer):
    """load dataset"""
    train_dataset_batch = BatchRetrievalDataset(tokenizer, max_len=config.max_seq_length, mode='train')
    train_dataset = RetrievalDataset(f"{config.data_path}/train.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=2, collate_fn=train_dataset_batch)
    dev_dataset_batch = BatchRetrievalDataset(tokenizer, max_len=config.max_seq_length, mode='val')
    dev_dataset = RetrievalDataset(f"{config.data_path}/dev.csv")
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=2, collate_fn=dev_dataset_batch)

    return train_dataloader, dev_dataloader

def build_index(all_embeddings, output_emb_size, hnsw_max_elements, hnsw_ef, hnsw_m):
    """"""
    index = hnswlib.Index(space="cosine", dim=output_emb_size if output_emb_size > 0 else 768)
    # Initializing index
    # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
    # during insertion of an element.
    # The capacity can be increased by saving/loading the index, see below.
    #
    # ef_construction - controls index search speed/build speed tradeoff
    #
    # M - is tightly connected with internal dimensionality of the data. Strongly affects memory consumption (~M)
    # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
    index.init_index(max_elements=hnsw_max_elements, ef_construction=hnsw_ef, M=hnsw_m)

    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    index.set_ef(hnsw_ef)

    # Set number of threads used during batch search/construction
    # By default using all available cores
    index.set_num_threads(16)
    print("start build index..........")
    index.add_items(all_embeddings)
    print("Total index number:{}".format(index.get_current_count()))
    return index

def get_label_data():
    """label data"""
    label_df = pd.read_csv(f"{config.data_path}/label.csv")
    label_df = label_df.dropna()
    id2label = {}
    label2id = {}
    for ind, row in label_df.iterrows():
        id2label[ind] = row['label']
        label2id[row['label']] = ind
    return id2label, label2id, label_df

def label_embedding(label_df, model, tokenizer):
    """"""
    all_embeddings = []
    for _, row in label_df.iterrows():
        label_source = tokenizer([row['label']], max_length=config.max_seq_length,
                  truncation=True, padding=True, return_tensors='pt')
        label_token = label_source.get('input_ids').squeeze(1).to(device)
        label_mask = label_source.get('attention_mask').squeeze(1).to(device)
        label_type_ids = label_source.get('token_type_ids').squeeze(1).to(device)
        label_emb = model.get_pooled_embedding(label_token, label_type_ids, label_mask)

        all_embeddings.append(label_emb.detach().cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings

def evaluation(model, tokenizer, valid_dataloader, label_data, label2id):
    """evaluation"""
    model.eval()
    all_embeddings = label_embedding(label_data, model, tokenizer)
    # print(len(all_embeddings))
    # print(all_embeddings[0])
    final_index = build_index(
        all_embeddings,
        output_emb_size=config.output_emb_size,
        hnsw_max_elements=config.hnsw_max_elements,
        hnsw_ef=config.hnsw_ef,
        hnsw_m=config.hnsw_m,
    )
    pred_label = {"y_label": [], "pred_label": []}

    for ind, batch_data in enumerate(valid_dataloader):
        text_token = batch_data['text_token'].to(device)
        text_type_ids = batch_data['text_type_ids'].to(device)
        text_mask = batch_data['text_mask'].to(device)
        query_embedding = model.get_pooled_embedding(text_token, text_type_ids, text_mask)
        recalled_idx, cosine_sims = final_index.knn_query(query_embedding.detach().cpu().numpy(), config.recall_num)
        batch_label = batch_data['batch_label']
        for i, recall in enumerate(recalled_idx):
            p_label = np.zeros(len(label2id))
            b_label = np.zeros(len(label2id))
            for j, rl in enumerate(recall):
                if cosine_sims[i][j] <= config.threshold:
                    p_label[rl] = 1
            b_list = list(batch_label[i].split(','))
            b_list = [label2id[bl] for bl in b_list]
            for bl in b_list:
                b_label[bl] = 1

            pred_label['y_label'].append(b_label)
            pred_label['pred_label'].append(p_label)

    micro_f1_score = f1_score(y_pred=pred_label['y_label'], y_true=pred_label['pred_label'], average="micro")

    return micro_f1_score


def train(load_model=False):
    """train model"""
    tokenizer = load_tokenizer(config.model_path)
    train_dataloader, dev_dataloader = get_train_dataloader(tokenizer)
    id2label, label2id, label_df = get_label_data()
    ptm_config = AutoConfig.from_pretrained(config.model_path)
    ptm = AutoModel.from_pretrained(config.model_path, ptm_config)
    model = SemanticIndexBatchNeg(
        ptm,
        ptm_config,
        device,
        pooling=config.pooling,
        dropout=config.dropout,
        margin=config.margin,
        scale=config.scale,
        output_emb_size=config.output_emb_size
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
    # optimizer = AdamW(
    #     model.parameters(),
    #     lr=float(config.learning_rate),
    #     weight_decay=0.01
    # )
    model.train()

    total_batch = 0
    best_loss = 0.00001

    start_time = time.time()
    for epoch in range(config.epochs):
        tqdm_bar = tqdm(train_dataloader, desc="training epoch{epoch}".format(epoch=epoch))
        loss_total, loss_diffs = [], []
        epoch_time = time.time()
        early_stop_batch = 0
        for batch_idx, batch_data in enumerate(tqdm_bar):
            text_token = batch_data['text_token'].to(device)
            text_type_ids = batch_data['text_type_ids'].to(device)
            text_mask = batch_data['text_mask'].to(device)
            label_token = batch_data['label_token'].to(device)
            label_type_ids = batch_data['label_type_ids'].to(device)
            label_mask = batch_data['label_mask'].to(device)
            # loss
            loss = model(text_token, label_token, text_type_ids, text_mask, label_type_ids, label_mask)

            # print(loss)
            # micro_f1_score = evaluation(model, tokenizer, dev_dataloader, label_df, label2id)
            # print(micro_f1_score)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_total.append(loss.detach().item())
            if batch_idx != 0:
                loss_diffs.append(abs(loss.detach().item() - loss_total[-2]))
            if total_batch % 1000 == 0 and total_batch != 0:
                micro_f1_score = evaluation(model, tokenizer, dev_dataloader, label_df, label2id)
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

def label_index(tokenizer):
    """"""
    id2label, label2id, label_df = get_label_data()
    ptm_config = AutoConfig.from_pretrained(config.model_path)
    ptm = AutoModel.from_pretrained(config.model_path, ptm_config)
    model = SemanticIndexBatchNeg(
        ptm,
        ptm_config,
        device,
        pooling=config.pooling,
        dropout=config.dropout,
        margin=config.margin,
        scale=config.scale,
        output_emb_size=config.output_emb_size
    )
    # model path
    best_model_path = f"{config.best_model}.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    model.eval()
    all_embeddings = label_embedding(label_df, model, tokenizer)
    final_index = build_index(
        all_embeddings,
        output_emb_size=config.output_emb_size,
        hnsw_max_elements=config.hnsw_max_elements,
        hnsw_ef=config.hnsw_ef,
        hnsw_m=config.hnsw_m,
    )
    final_index.save_index("bert_index.bin")
    del final_index

def predict(text, tokenizer):
    """predict"""
    # load model
    ptm_config = AutoConfig.from_pretrained(config.model_path)
    ptm = AutoModel.from_pretrained(config.model_path, ptm_config)
    model = SemanticIndexBatchNeg(
        ptm,
        ptm_config,
        device,
        pooling=config.pooling,
        dropout=config.dropout,
        margin=config.margin,
        scale=config.scale,
        output_emb_size=config.output_emb_size
    )
    # model path
    best_model_path = f"{config.best_model}.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    model.eval()
    # load index
    index = hnswlib.Index(space="cosine", dim=config.output_emb_size if config.output_emb_size > 0 else 768)
    index.load_index("bert_index.bin", max_elements=config.hnsw_max_elements)
    # print(index.get_current_count())
    # load label
    id2label, label2id, label_df = get_label_data()
    t1 = time.time()
    label_source = tokenizer([text], max_length=config.max_seq_length,
                             truncation=True, padding=True, return_tensors='pt')
    text_token = label_source.get('input_ids').squeeze(1).to(device)
    text_mask = label_source.get('attention_mask').squeeze(1).to(device)
    text_type_ids = label_source.get('token_type_ids').squeeze(1).to(device)
    # print(input_ids)
    text_embedding = model.get_pooled_embedding(text_token, text_type_ids, text_mask)

    idx, distances = index.knn_query(text_embedding.detach().numpy(), config.recall_num)
    for i, ind in enumerate(idx[0]):
        label = id2label[ind]
        distance = distances[0][i]
        print(label, distance)
    print(time.time() - t1)


if __name__ == '__main__':
    """
    数据集：链接: https://pan.baidu.com/s/1n5RsF5y-1HUGbm6GCv76hg?pwd=rdxx 提取码: rdxx
    """
    tokenizer = load_tokenizer(config.model_path)
    # train(load_model=False)
    # label_index(tokenizer)
    text = '十个月婴儿的能力.刚满十个月婴儿应具备哪些能力?'
    print(text)
    predict(text, tokenizer)
