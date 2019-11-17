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
from model.model_util import InitType
from model.model_util import init_tensor


class RegionEmbedding(Classifier):
    """Implement region embedding classification method
    Reference: "A New Method of Region Embedding for Text Classification"
    """

    def __init__(self, dataset, config):
        super(RegionEmbedding, self).__init__(dataset, config)
        self.region_size = config.embedding.region_size
        self.radius = int(self.region_size / 2)
        self.linear = torch.nn.Linear(config.embedding.dimension,
                                      len(dataset.label_map))
        init_tensor(self.linear.weight, init_type=InitType.XAVIER_UNIFORM)
        init_tensor(self.linear.bias, init_type=InitType.UNIFORM, low=0, high=0)

    def get_parameter_optimizer_dict(self):
        params = super(RegionEmbedding, self).get_parameter_optimizer_dict()
        return params

    def forward(self, batch):
        embedding, _, mask = self.get_embedding(
            batch, [self.radius, self.radius], cDataset.VOCAB_PADDING)
        # mask should have same dim with padded embedding
        embedding = self.token_similarity_attention(embedding)
        mask = torch.nn.functional.pad(mask, (self.radius, self.radius, 0, 0), "constant", 0)
        mask = mask.unsqueeze(2)
        embedding = embedding * mask
        doc_embedding = torch.sum(embedding, 1)
        doc_embedding = self.dropout(doc_embedding)
        return self.linear(doc_embedding)

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
