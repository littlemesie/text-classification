# -*- coding:utf-8 -*-

"""
@date: 2023/11/17 下午2:20
@summary:
"""
import time
import torch
import pandas as pd
import onnxruntime as ort
from modelscope.pipelines import pipeline
from modelscope.exporters import Exporter
from modelscope.models import Model
from modelscope.preprocessors import Preprocessor
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline as pl
import tempfile
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.hub import read_config

tokenizer_path = "/home/mesie/apache/nlp_structbert_zero-shot-classification_chinese-large"
# tokenizer_path = "/home/mesie/apache/nlp_mt5_zero-shot-augment_chinese-base"

dispute_label_ind = {
    "negative": 0, "矛盾纠纷": 1, "人际纠纷": 2, "治安纠纷": 3, "打架纠纷": 4, "操场纠纷": 5, "出租车纠纷": 6, "中介纠纷": 7,
    "医托纠纷": 8, "假币纠纷": 9, "家暴纠纷": 10, "教育问题纠纷": 11, "退役军人问题纠纷": 12, "家庭婚姻感情纠纷": 13,
    "经济纠纷": 14, "邻里纠纷": 15, "劳资纠纷": 16, "医疗纠纷": 17
}
# labels = ['自杀', '纠纷', '家暴', '咨询']
labels = [k for k, v in dispute_label_ind.items()]
labels.remove('negative')

# model = Model.from_pretrained(tokenizer_path)
# output_files = Exporter.from_model(model).export_onnx(
#     candidate_labels=labels,
#     hypothesis_template='标签是{}',
#     output_dir='tmp')
# print(output_files)

sentence = '经民警现场了解处理，两口子感情不和，男方长期精力暴力控制女方，已转社会跟踪协调'
# preprocessor = Preprocessor.from_pretrained(tokenizer_path)
# ort_session = ort.InferenceSession('tmp/model.onnx')
# t11 = time.time()
# outputs = ort_session.run(['logits'], dict(preprocessor(sentence,
#                                            candidate_labels=labels,
#                                            hypothesis_template='标签是{}', return_tensors='np')))
# print(time.time() - t11)
# print(outputs)
#
# classifier = pipeline('zero-shot-classification', tokenizer_path)
# # classifier = pipeline('text2text-generation', tokenizer_path)
#
# t1 = time.time()
# res = classifier(sentence, candidate_labels=labels)
# print(res)
# print(time.time() - t1)
# print(f"文本分类。\n候选标签：{','.join(labels)}。\n文本内容：{sentence}")
# t1 = time.time()
# print(classifier(f"文本分类。\n候选标签：{','.join(labels)}。\n文本内容：{sentence}"))
# print(time.time() - t1)

# model = AutoModelForTokenClassification.from_pretrained(tokenizer_path)
# print(model)
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# print(tokenizer)
# classifier = pl('zero-shot-classification', tokenizer_path)
# t1 = time.time()
# res = classifier(sentence, candidate_labels=labels)
# print(res)
# print(time.time() - t1)