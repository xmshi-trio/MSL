#!usr/bin/env python
# coding:utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
"""

import torch

from dataset.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.layers import SumAttention
from model.rnn import RNN
from util import Type


class DocEmbeddingType(Type):
    """Standard names for doc embedding type.
    """
    AVG = 'AVG'
    ATTENTION = 'Attention'
    LAST_HIDDEN = 'LastHidden'

    @classmethod
    def str(cls):
        return ",".join(
            [cls.AVG, cls.ATTENTION, cls.LAST_HIDDEN])


class TextRNN(Classifier):
    """Implement TextRNN, contains LSTM，BiLSTM，GRU，BiGRU
    Reference: "Effective LSTMs for Target-Dependent Sentiment Classification"
               "Bidirectional LSTM-CRF Models for Sequence Tagging"
               "Generative and discriminative text classification
                with recurrent neural networks"
    """

    def __init__(self, dataset, config):
        super(TextRNN, self).__init__(dataset, config)
        self.rnn = RNN(
            config.embedding.dimension, config.TextRNN.hidden_dimension,
            num_layers=config.TextRNN.num_layers, batch_first=True,
            bidirectional=config.TextRNN.bidirectional,
            rnn_type=config.TextRNN.rnn_type)
        self.rnn2 = RNN(
            config.embedding.dimension, config.TextRNN.hidden_dimension,
            num_layers=config.TextRNN.num_layers, batch_first=True,
            bidirectional=config.TextRNN.bidirectional,
            rnn_type=config.TextRNN.rnn_type)
        hidden_dimension = config.TextRNN.hidden_dimension
        if config.TextRNN.bidirectional:
            hidden_dimension *= 2
        self.sum_attention = SumAttention(hidden_dimension,
                                          config.TextRNN.attention_dimension,
                                          config.device)
        self.linear = torch.nn.Linear(hidden_dimension, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = super(TextRNN, self).get_parameter_optimizer_dict()
        params.append({'params': self.rnn.parameters()})
        params.append({'params': self.linear.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0.0

    def forward(self, batch):
        if self.config.feature.feature_names[0] == "token":
            embedding = self.token_embedding(
                batch[cDataset.DOC_TOKEN].to(self.config.device))
            length = batch[cDataset.DOC_TOKEN_LEN].to(self.config.device)
        else:
            embedding = self.char_embedding(
                batch[cDataset.DOC_CHAR].to(self.config.device))
            length = batch[cDataset.DOC_CHAR_LEN].to(self.config.device)
        #output1, _ = self.rnn2(embedding, length)
        #embedding = self.token_similarity_attention(embedding)
        output, last_hidden = self.rnn(embedding, length)
        
        doc_embedding_type = self.config.TextRNN.doc_embedding_type
        if doc_embedding_type == DocEmbeddingType.AVG:
            doc_embedding = torch.sum(output, 1) / length.unsqueeze(1)
        elif doc_embedding_type == DocEmbeddingType.ATTENTION:
            doc_embedding = self.sum_attention(output)
        elif doc_embedding_type == DocEmbeddingType.LAST_HIDDEN:
            doc_embedding = last_hidden
        else:
            raise TypeError(
                "Unsupported rnn init type: %s. Supported rnn type is: %s" % (
                    doc_embedding_type, DocEmbeddingType.str()))

        return self.dropout(self.linear(doc_embedding))

    def token_similarity_attention(self, output):
        # output: (batch, sentence length, embedding dim)
        symptom_id_list = [6, 134, 15, 78, 2616, 257, 402, 281, 14848, 71, 82, 96, 352, 60, 227, 204, 178, 175, 233, 192, 416, 91, 232, 317, 17513, 628, 1047]
        symptom_embedding = self.token_embedding(torch.LongTensor(symptom_id_list).cuda())
        # symptom_embedding: torch.tensor(symptom_num, embedding dim)
        batch_symptom_embedding = torch.cat([symptom_embedding.view(1, symptom_embedding.shape[0], -1)] * output.shape[0], dim=0)
        similarity = torch.sigmoid(torch.bmm(torch.nn.functional.normalize(output, dim=2), torch.nn.functional.normalize(batch_symptom_embedding.permute(0, 2, 1), dim=2)))
        #similarity = torch.bmm(torch.nn.functional.normalize(output, dim=2), torch.nn.functional.normalize(batch_symptom_embedding.permute(0, 2, 1), dim=2))
        #similarity = torch.sigmoid(torch.max(similarity, dim=2)[0])
        similarity = torch.max(similarity, dim=2)[0]
        #similarity = torch.sigmoid(torch.sum(similarity, dim=2))
        # similarity: torch.tensor(batch, sentence_len)
        similarity = torch.cat([similarity.view(similarity.shape[0], -1, 1)] * output.shape[2], dim=2)
        # similarity: torch.tensor(batch, batch, sentence_len, embedding dim)
        #sentence_embedding = torch.sum(torch.mul(similarity, output), dim=1)
        # sentence_embedding: (batch, embedding)
        sentence_embedding = torch.mul(similarity, output)
        # sentence_embedding: (batch, sentence len, embedding)
        return sentence_embedding