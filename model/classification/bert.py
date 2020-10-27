#!usr/bin/env python
# coding:utf-8
import torch

from dataset.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from transformers import *

class BERT(Classifier):
    def __init__(self, dataset, config):
        super(BERT, self).__init__(dataset, config)
        self.bert_model = BertModel.from_pretrained(config.data.pretrained_bert_embedding)
        self.linear = torch.nn.Linear(config.embedding.dimension, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def forward(self, batch):
        embedding = self.bert_model(torch.LongTensor(batch['doc_token']).to(self.config.device))[1]
        return self.dropout(self.linear(embedding))
