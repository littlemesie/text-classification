# -*- coding:utf-8 -*-

"""
@date: 2023/11/17 下午2:20
@summary:
"""
import time
import torch
import pandas as pd
import numpy as np
import onnxruntime as ort
from collections import OrderedDict
from modelscope.pipelines import pipeline
from modelscope.exporters import Exporter
from modelscope.models import Model
from modelscope import AutoTokenizer, AutoModel
from modelscope.exporters import TorchModelExporter
from modelscope.utils.constant import Tasks
from modelscope.preprocessors import Preprocessor
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, BertForSequenceClassification
from transformers import pipeline as pl
import tempfile
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.hub import read_config

# tokenizer_path = "/media/mesie/a1d6502f-8a4a-4017-a9b5-3777bd223927/model/modelscope/structbert/nlp_gte_sentence-embedding_chinese-large"
#
tokenizer_path = "/media/mesie/a1d6502f-8a4a-4017-a9b5-3777bd223927/model//modelscope/nlp_structbert_zero-shot-classification_chinese-base"
#
# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)


# model = BertModel.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(tokenizer_path, num_labels=3)
# output_files = Exporter.from_model(model).export_onnx(output_dir='tmp')
# print(output_files)
#
# text1 = '一人报警，看见一个人从嘉陵江长江大桥跳下去了'
# text1 = '商务职业学院和财经职业学院哪个好？'
# text2 = '商务职业学院和财经职业学院哪个好？'
# text2 = '商务职业学院商务管理在哪个校区？'
# text2 = '看见一个人从嘉陵江长江大桥跳下去了'
# t1 = time.time()
# inputs1 = tokenizer(text1, return_tensors="pt")

# print(inputs1)
# inputs2 = tokenizer(text2, return_tensors="pt")
#
# outputs1 = model(**inputs1)
# print(time.time() - t1)
# outputs2 = model(**inputs2)
# print(outputs1.last_hidden_state[:, 0].shape)
# print(torch.nn.functional.cosine_similarity(outputs1.last_hidden_state[:, 0], outputs2.last_hidden_state[:, 0]))

# export model
# dummy_inputs = tokenizer(tokenizer.unk_token, padding='max_length', max_length=512, return_tensors='pt')
# print(dummy_inputs)
# dynamic_axis = {0: 'batch', 1: 'sequence'}
# inputs = OrderedDict([
#     ('input_ids', dynamic_axis),
#     ('attention_mask', dynamic_axis),
#     ('token_type_ids', dynamic_axis),
# ])
# outputs = OrderedDict({'output': {0: 'batch'}})
# torch.onnx.export(
#                  model,
#                 (dummy_inputs["input_ids"],
#                  dummy_inputs["attention_mask"],
#                  dummy_inputs["token_type_ids"]),
#                 './tmp/model.onnx',
#                 opset_version=15,
#                 do_constant_folding=True,
#                 input_names=["input_ids", "attention_mask", "token_type_ids"],
#                 output_names=["output"],
#                 dynamic_axes=inputs
#             )



# ort_session = ort.InferenceSession('./tmp/model.onnx')
# t1 = time.time()
# inputs1 = tokenizer(text1, return_tensors="pt")
# # print(inputs1)
# output1 = ort_session.run(None, {'input_ids': inputs1["input_ids"].detach().numpy(),
#                                 'attention_mask': inputs1['attention_mask'].detach().numpy(),
#                                 'token_type_ids': inputs1['token_type_ids'].detach().numpy()})
# print(time.time() - t1)
# inputs2 = tokenizer(text2, return_tensors="pt")
# output2 = ort_session.run(None, {'input_ids': inputs2["input_ids"].detach().numpy(),
#                                 'attention_mask': inputs2['attention_mask'].detach().numpy(),
#                                 'token_type_ids': inputs2['token_type_ids'].detach().numpy()})
#
#
# print(torch.nn.functional.cosine_similarity(torch.Tensor(output1[0][:, 0]), torch.Tensor(output2[0][:, 0])))
# print(time.time() - t1)
# print(output1[0][:, 0].shape)

text = '两车相撞，一人受伤[SEP]受伤'
t1 = time.time()
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
print(time.time() -t1)
print(outputs)
from modelscope.pipelines import pipeline

# model = AutoModelForTokenClassification.from_pretrained(tokenizer_path)
# print(model)
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# print(tokenizer)
# classifier = pl('zero-shot-classification', tokenizer_path)
# t1 = time.time()
# res = classifier(sentence, candidate_labels=labels)
# print(res)
# print(time.time() - t1)