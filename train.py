#!/usr/bin/env python
#coding:utf-8
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

import os
import shutil
import sys
import time
import random

import torch
from torch.utils.data import DataLoader

import util
import numpy as np
from config import Config
from dataset.classification_dataset import ClassificationDataset
from dataset.collator import ClassificationCollator
from dataset.collator import FastTextCollator
from dataset.collator import ClassificationType
from evaluate.classification_evaluate import \
    ClassificationEvaluator as cEvaluator
from model.classification.drnn import DRNN
from model.classification.fasttext import FastText
from model.classification.textcnn import TextCNN
from model.classification.textvdcnn import TextVDCNN
from model.classification.textrnn import TextRNN
from model.classification.textrcnn import TextRCNN
from model.classification.transformer import Transformer
from model.classification.dpcnn import DPCNN
from model.classification.attentive_convolution import AttentiveConvNet
from model.classification.region_embedding import RegionEmbedding
from model.classification.bert import BERT
from model.loss import ClassificationLoss
from model.model_util import get_optimizer, get_hierar_relations
from util import ModeType
from transformers import *


ClassificationDataset, ClassificationCollator, FastTextCollator, ClassificationLoss, cEvaluator
FastText, TextCNN, TextRNN, TextRCNN, DRNN, TextVDCNN, Transformer, DPCNN, AttentiveConvNet, RegionEmbedding, BERT


def get_data_loader(dataset_name, collate_name, conf):
    """Get data loader: Train, Validate, Test
    """
    train_dataset = globals()[dataset_name](
        conf, conf.data.train_json_files, generate_dict=True)
    collate_fn = globals()[collate_name](conf, len(train_dataset.label_map))
    train_data_loader = DataLoader(
        train_dataset, batch_size=conf.train.batch_size, shuffle=True,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    validate_dataset = globals()[dataset_name](
        conf, conf.data.validate_json_files)
    validate_data_loader = DataLoader(
        validate_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    test_dataset = globals()[dataset_name](conf, conf.data.test_json_files)
    test_data_loader = DataLoader(
        test_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    return train_data_loader, validate_data_loader, test_data_loader


def get_unlabeled_data_loader(dataset_name, collate_name, conf):
    """Get unlabeled data loader: Train
    """
    unlabeled_train_dataset = globals()[dataset_name](conf, conf.data.unlabeled_train_json_files, generate_dict=False)
    collate_fn = globals()[collate_name](conf, len(unlabeled_train_dataset.label_map))
    unlabeled_train_data_loader = DataLoader(unlabeled_train_dataset, batch_size=conf.train.batch_size, shuffle=True, num_workers=conf.data.num_worker, collate_fn=collate_fn, pin_memory=True)

    unlabeled_dev_dataset = globals()[dataset_name](conf, conf.data.unlabeled_dev_json_files, generate_dict=False)
    collate_fn = globals()[collate_name](conf, len(unlabeled_dev_dataset.label_map))
    unlabeled_dev_data_loader = DataLoader(unlabeled_dev_dataset, batch_size=conf.train.batch_size, shuffle=True, num_workers=conf.data.num_worker, collate_fn=collate_fn, pin_memory=True)
    
    unlabeled_test_dataset = globals()[dataset_name](conf, conf.data.unlabeled_test_json_files, generate_dict=False)
    collate_fn = globals()[collate_name](conf, len(unlabeled_test_dataset.label_map))
    unlabeled_test_data_loader = DataLoader(unlabeled_test_dataset, batch_size=conf.train.batch_size, shuffle=True, num_workers=conf.data.num_worker, collate_fn=collate_fn, pin_memory=True)
    return unlabeled_train_data_loader, unlabeled_dev_data_loader, unlabeled_test_data_loader


def get_classification_model(model_name, dataset, conf):
    """Get classification model from configuration
    """
    model = globals()[model_name](dataset, conf)
    model = model.cuda(conf.device) if conf.device.startswith("cuda") else model
    return model


class ClassificationTrainer(object):
    def __init__(self, label_map, logger, evaluator, conf, loss_fn):
        self.label_map = label_map
        self.logger = logger
        self.evaluator = evaluator
        self.conf = conf
        self.loss_fn = loss_fn
        if self.conf.task_info.hierarchical:
            self.hierar_relations = get_hierar_relations(
                    self.conf.task_info.hierar_taxonomy, label_map)

    def train(self, data_loader, model, optimizer, stage, epoch):
        model.update_lr(optimizer, epoch)
        model.train()
        return self.run(data_loader, model, optimizer, stage, epoch,
                        ModeType.TRAIN)

    def eval(self, data_loader, model, optimizer, stage, epoch):
        model.eval()
        return self.run(data_loader, model, optimizer, stage, epoch)

    def run(self, data_loader, model, optimizer, stage,
            epoch, mode=ModeType.EVAL):
        is_multi = False
        # multi-label classifcation
        if self.conf.task_info.label_type == ClassificationType.MULTI_LABEL:
            is_multi = True
        predict_probs = []
        standard_labels = []
        num_batch = data_loader.__len__()
        total_loss = 0.
        
        if self.conf.model_name != "BERT":
            for batch in data_loader.dataset:
                logits = model(batch)
                # hierarchical classification
                if self.conf.task_info.hierarchical:
                    linear_paras = model.linear.weight
                    is_hierar = True
                    used_argvs = (self.conf.task_info.hierar_penalty, linear_paras, self.hierar_relations)
                    loss = self.loss_fn(
                        logits,
                        batch[ClassificationDataset.DOC_LABEL].to(self.conf.device),
                        is_hierar,
                        is_multi,
                        *used_argvs)
                else:  # flat classification
                    loss = self.loss_fn(
                        logits,
                        batch[ClassificationDataset.DOC_LABEL].to(self.conf.device))
                if mode == ModeType.TRAIN:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    continue
                total_loss += loss.item()
                if not is_multi:
                    result = torch.nn.functional.softmax(logits, dim=1).cpu().tolist()
                else:
                    result = torch.sigmoid(logits).cpu().tolist()
                predict_probs.extend(result)
                standard_labels.extend(batch[ClassificationDataset.DOC_LABEL_LIST])
        else:
            for batch in data_loader:
                logits = model(batch)
                # hierarchical classification
                if self.conf.task_info.hierarchical:
                    linear_paras = model.linear.weight
                    is_hierar = True
                    used_argvs = (self.conf.task_info.hierar_penalty, linear_paras, self.hierar_relations)
                    loss = self.loss_fn(
                        logits,
                        batch[ClassificationDataset.DOC_LABEL].to(self.conf.device),
                        is_hierar,
                        is_multi,
                        *used_argvs)
                else:  # flat classification
                    loss = self.loss_fn(
                        logits,
                        batch[ClassificationDataset.DOC_LABEL].to(self.conf.device))
                if mode == ModeType.TRAIN:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    continue
                total_loss += loss.item()
                if not is_multi:
                    result = torch.nn.functional.softmax(logits, dim=1).cpu().tolist()
                else:
                    result = torch.sigmoid(logits).cpu().tolist()
                predict_probs.extend(result)
                standard_labels.extend(batch[ClassificationDataset.DOC_LABEL_LIST])

        if mode == ModeType.EVAL:
            total_loss = total_loss / num_batch
            (_, precision_list, recall_list, fscore_list, right_list,
             predict_list, standard_list, turn_accuracy) = \
                self.evaluator.evaluate(
                    predict_probs, standard_label_ids=standard_labels, label_map=self.label_map,
                    threshold=self.conf.eval.threshold, top_k=self.conf.eval.top_k,
                    is_flat=self.conf.eval.is_flat, is_multi=is_multi)
            # precision_list[0] save metrics of flat classification
            # precision_list[1:] save metrices of hierarchical classification
            #self.logger.warn(
            self.logger.info(
                "%s performance at epoch %d is precision: %f, "
                "recall: %f, fscore: %f, macro-fscore: %f, right: %d, predict: %d, standard: %d.\n"
                "Turn accuracy: %f, Loss is: %f." % (
                    stage, epoch, precision_list[0][cEvaluator.MICRO_AVERAGE],
                    recall_list[0][cEvaluator.MICRO_AVERAGE],
                    fscore_list[0][cEvaluator.MICRO_AVERAGE],
                    fscore_list[0][cEvaluator.MACRO_AVERAGE],
                    right_list[0][cEvaluator.MICRO_AVERAGE],
                    predict_list[0][cEvaluator.MICRO_AVERAGE],
                        standard_list[0][cEvaluator.MICRO_AVERAGE], turn_accuracy, total_loss))
            return fscore_list[0][cEvaluator.MICRO_AVERAGE]

def load_checkpoint(file_name, conf, model, optimizer):
    checkpoint = torch.load(file_name)
    conf.train.start_epoch = checkpoint["epoch"]
    best_performance = checkpoint["best_performance"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return best_performance

def save_checkpoint(state, file_prefix):
    file_name = file_prefix + "_" + str(state["epoch"])
    torch.save(state, file_name)

def trans_id_to_bert_embedding(data_loader, model, conf):
    # data_loader: ClassificationDataset object
    output_dataset = []
    for batch in data_loader:
        temp_batch = batch
        with torch.no_grad():
            temp = model(batch['doc_token'])[0]
        temp_batch['doc_token'] = temp
        output_dataset.append(temp_batch)
    data_loader.dataset = output_dataset
    return data_loader

def trans_list2mat(label_list, shape):
    output_mat = torch.zeros(shape)
    for i in range(len(label_list)):
        for item in label_list[i]:
            output_mat[i][item] = 1.0
    return output_mat

def select_unlabeled_data(model, unlabeled_data_loader, label_size, conf):
    data_loader = unlabeled_data_loader
    output_dataset = []
    for batch in unlabeled_data_loader.dataset:
        temp_batch = dict()
        temp_batch = batch.copy()
        batch_size = batch['doc_token'].size()[0]
        logits = model(batch)
        if conf.task_info.top_n_teacher:
            pseudo_label = torch.topk(logits, k=conf.task_info.top_n_teacher, dim=1)[1].tolist()
        else:
            result = torch.sigmoid(logits).cpu().tolist()
            prob_np = np.array(result, dtype=np.float32)
            pseudo_label = []
            predict_label_idx = np.argsort(-prob_np, axis=1)
            #for i in range(conf.train.batch_size):
            for i in range(predict_label_idx.shape[0]):
                pseudo_label.append([predict_label_idx[i][j] for j in range(0, 5) if prob_np[i][predict_label_idx[i][j]]])
        weak_label = batch['doc_label_list']
        selected_label = []
        for i in range(len(weak_label)):
            temp_augmented_label = pseudo_label[i]
            if conf.task_info.add_noise:
                for j in range(len(temp_augmented_label)):
                    temp_augmented_label[j] = random.randint(0, 28)
            if conf.task_info.Augmentation_Method == "Union":
                selected_label.append(list(set(weak_label[i] + temp_augmented_label)))
            elif conf.task_info.Augmentation_Method == "Intersection":
                selected_label.append(list(set(weak_label[i])&set(temp_augmented_label)))
            elif conf.task_info.Augmentation_Method == "Self Learning":
                selected_label.append(temp_augmented_label)

        temp_batch['doc_label'] = trans_list2mat(selected_label, (batch_size, label_size))
        temp_batch['doc_label_list'] = selected_label
        output_dataset.append(temp_batch)
        output_dataset.append(batch)
    data_loader.dataset = output_dataset
    return data_loader

def get_data(conf):
    bert_model = BertModel.from_pretrained(conf.data.pretrained_bert_embedding)
    # logger = util.Logger(conf)
    if not os.path.exists(conf.checkpoint_dir):
        os.makedirs(conf.checkpoint_dir)

    model_name = conf.model_name
    dataset_name = "ClassificationDataset"
    collate_name = "FastTextCollator" if model_name == "FastText" else "ClassificationCollator"
    collate_name = "FastTextCollator" if model_name == "FastText" \
        else "ClassificationCollator"
    global train_data_loader
    global validate_data_loader
    global test_data_loader
    global unlabeled_train_data_loader
    global unlabeled_validate_data_loader
    global unlabeled_test_data_loader
    global empty_dataset

    train_data_loader, validate_data_loader, test_data_loader = \
        get_data_loader(dataset_name, collate_name, conf)
    if conf.model_name != "BERT":
        train_data_loader = trans_id_to_bert_embedding(train_data_loader, bert_model, conf)
        validate_data_loader = trans_id_to_bert_embedding(validate_data_loader, bert_model, conf)
        test_data_loader = trans_id_to_bert_embedding(test_data_loader, bert_model, conf)
    if conf.task_info.weak_pretrain:
        unlabeled_train_data_loader, unlabeled_validate_data_loader, unlabeled_test_data_loader = \
            get_unlabeled_data_loader(dataset_name, collate_name, conf)
        if conf.model_name != "BERT":
            unlabeled_train_data_loader = trans_id_to_bert_embedding(unlabeled_train_data_loader, bert_model, conf)
            unlabeled_validate_data_loader = trans_id_to_bert_embedding(unlabeled_validate_data_loader, bert_model, conf)
            unlabeled_test_data_loader = trans_id_to_bert_embedding(unlabeled_test_data_loader, bert_model, conf)
    empty_dataset = globals()[dataset_name](conf, [])


def train(conf): 
    model_name = conf.model_name
    logger = util.Logger(conf)
    if conf.task_info.weak_pretrain:
        logger.info("Batch Size: " + str(conf.train.batch_size) + " Pretrain Num Epoch: " + str(conf.train.pretrain_num_epochs))
    else:
        logger.info("Batch Size: " + str(conf.train.batch_size))

    if conf.task_info.weak_pretrain and conf.task_info.weak_data_augmentation:
        model_teacher = get_classification_model(model_name, empty_dataset, conf)
        if conf.model_name != "BERT":
            optimizer_teacher = get_optimizer(conf, model_teacher)
        else:
            optimizer_teacher = AdamW(model_teacher.parameters(), lr = 5e-2, eps = 1e-2)
        # optimizer_teacher: optimizer for teacher model

    model_target = get_classification_model(model_name, empty_dataset, conf)
    loss_fn = globals()["ClassificationLoss"](
        label_size=len(empty_dataset.label_map), loss_type=conf.train.loss_type)

    if conf.task_info.weak_pretrain:
        if conf.model_name != "BERT":
            optimizer_weak = get_optimizer(conf, model_target)
        else:
            optimizer_weak = AdamW(model_target.parameters(), lr = 5e-2, eps = 1e-2)
        # optimizer_weak: optimizer for target model pretraining stage
    if conf.model_name != "BERT":
        optimizer_target = get_optimizer(conf, model_target)
    else:
        optimizer_target = AdamW(model_target.parameters(), lr = 5e-2, eps = 1e-2)
    # optimizer_target: optimizer for target model fine-tuning stage
    evaluator = cEvaluator(conf.eval.dir)
    
    trainer_target = globals()["ClassificationTrainer"](
        empty_dataset.label_map, logger, evaluator, conf, loss_fn)
    # trainer_target: trainer for target model on fine-tuning stage
    if conf.task_info.weak_pretrain:
        trainer_weak = globals()["ClassificationTrainer"](
            empty_dataset.label_map, logger, evaluator, conf, loss_fn)
        # trainer_weak: trainer for target model on pretraining stage
        if conf.task_info.weak_data_augmentation:
            trainer_teacher = globals()["ClassificationTrainer"](
                empty_dataset.label_map, logger, evaluator, conf, loss_fn)
            # trainer_teacher: trainer for teacher model
    
    if conf.task_info.weak_data_augmentation:
        best_epoch = -1
        best_performance = 0
        model_file_prefix = conf.checkpoint_dir + "/" + model_name + "_teacher"

        logger.info("Training Teacher Model on Labeled Data")
        for epoch in range(conf.train.start_epoch, conf.train.start_epoch + conf.train.num_epochs):
            start_time = time.time()
            trainer_teacher.train(train_data_loader, model_teacher, optimizer_teacher, "Train", epoch)
            trainer_teacher.eval(train_data_loader, model_teacher, optimizer_teacher, "Train", epoch)
            performance = trainer_teacher.eval(
                validate_data_loader, model_teacher, optimizer_teacher, "Validate", epoch)
            trainer_teacher.eval(test_data_loader, model_teacher, optimizer_teacher, "Test", epoch)

            if performance > best_performance:  # record the best model
                best_epoch = epoch
                best_performance = performance
                temp_model = model_teacher
                save_checkpoint({
                    'epoch': epoch,
                    'model_name': model_name,
                    'state_dict': model_teacher.state_dict(),
                    'best_performance': best_performance,
                    'optimizer': optimizer_teacher.state_dict(),
                }, model_file_prefix)

            time_used = time.time() - start_time
            logger.info("Epoch %d cost time: %d second" % (epoch, time_used))
    best_epoch = -1
    best_performance = 0
    if conf.task_info.weak_pretrain:        
        if conf.task_info.weak_data_augmentation:
            unlabeled_data_train_data_loader = select_unlabeled_data(temp_model, unlabeled_train_data_loader, len(trainer_weak.label_map), conf)
        
        logger.info("Pretraining on Weak Supervision Data")
        for epoch in range(conf.train.start_epoch,
                       conf.train.start_epoch + conf.train.pretrain_num_epochs):
            start_time = time.time()
            trainer_weak.train(unlabeled_train_data_loader, model_target, optimizer_weak, "Train", epoch)
            trainer_weak.eval(unlabeled_train_data_loader, model_target, optimizer_weak, "Train", epoch)
            performance = trainer_weak.eval(
                validate_data_loader, model_target, optimizer_weak, "Validate", epoch)
            trainer_weak.eval(test_data_loader, model_target, optimizer_weak, "Test", epoch)
            
            if performance > best_performance:  # record the best model
                temp_model = model_target
            time_used = time.time() - start_time
            logger.info("Epoch %d cost time: %d second" % (epoch, time_used))
        model_target = temp_model

    logger.info("Fine-tuning on Labeled Data")
        
    best_epoch = -1
    best_performance = 0
    if conf.task_info.weak_pretrain:
        if conf.task_info.weak_data_augmentation:
            model_file_prefix = conf.checkpoint_dir + "/" + model_name + "-Augmentation-" + conf.task_info.Augmentation_Method + "-Pretrain" + str(conf.train.pretrain_num_epochs) + "-Batch" + str(conf.train.batch_size)
        else:
            model_file_prefix = conf.checkpoint_dir + "/" + model_name + "-WeakSupervision-" + "-Pretrain" + str(conf.train.pretrain_num_epochs) + "-Batch" + str(conf.train.batch_size)
    else:
        model_file_prefix = conf.checkpoint_dir + "/" + model_name + "-Batch" + str(conf.train.batch_size)
    for epoch in range(conf.train.start_epoch,
                   conf.train.start_epoch + conf.train.num_epochs):
        start_time = time.time()
        trainer_target.train(train_data_loader, model_target, optimizer_target, "Train", epoch)
        trainer_target.eval(train_data_loader, model_target, optimizer_target, "Train", epoch)
        performance = trainer_target.eval(validate_data_loader, model_target, optimizer_target, "Validate", epoch)
        trainer_target.eval(test_data_loader, model_target, optimizer_target, "Test", epoch)
        if performance > best_performance:  # record the best model
            best_epoch = epoch
            best_performance = performance
            temp_model = model_target
            save_checkpoint({
                    'epoch': epoch,
                    'model_name': model_name,
                    'state_dict': model_target.state_dict(),
                    'best_performance': best_performance,
                    'optimizer': optimizer_target.state_dict(),
                    }, model_file_prefix)
        time_used = time.time() - start_time
        logger.info("Epoch %d cost time: %d second" % (epoch, time_used))
    
    logger.info("The Best Performance on Validation Data and Test Data")  
    #best_epoch_file_name = model_file_prefix + "_" + str(best_epoch)
    #best_file_name = model_file_prefix + "_best"
    #shutil.copyfile(best_epoch_file_name, best_file_name)
    #load_checkpoint(model_file_prefix + "_" + str(best_epoch), conf, model,
    #                optimizer)
    model = temp_model
    trainer_target.eval(train_data_loader, model, optimizer_target, "Best Train", best_epoch)
    trainer_target.eval(validate_data_loader, model, optimizer_target, "Best Validate", best_epoch)
    trainer_target.eval(test_data_loader, model, optimizer_target, "Best Test", best_epoch)

if __name__ == '__main__':
    config = Config(config_file=sys.argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.train.visible_device_list)
    torch.manual_seed(2019)
    torch.cuda.manual_seed(2019)
    np.random.seed(2019)
    for batch_size in config.train.batch_size:
        conf = Config(config_file=sys.argv[1])
        conf.train.batch_size = batch_size
        get_data(conf)
        if conf.task_info.weak_pretrain:
            for pretrain_num in config.train.pretrain_num_epochs:
                conf1 = Config(config_file=sys.argv[1])
                conf1.train.batch_size = batch_size
                conf1.train.pretrain_num_epochs = pretrain_num
                train(conf1)
        else:
            conf1 = Config(config_file=sys.argv[1])
            conf1.train.batch_size = batch_size
            train(conf1)
