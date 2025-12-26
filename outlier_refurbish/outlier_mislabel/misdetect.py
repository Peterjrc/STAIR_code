import random
import math
import torch
import pandas as pd
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import time
import os
import argparse
import torch.nn.functional as F
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.preprocessing import Normalizer, RobustScaler, MinMaxScaler
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from utils import *
from model import Model, Contrastive_Model, Reconstruct_Model
from run_detector import run_detector


def inference_NN(args, model, eval_loader, outlier_ratio):

    model.eval()
    logits = []
    confidences = []
    pred_labels = []

    for idx, data, label in eval_loader:
        data, label = data.cuda(), label.cuda()
        pred_logits = model(data)
        pred_prob = torch.softmax(pred_logits, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1)
        logits.append(pred_logits)
        confidences.append(pred_prob[:, 1])
        pred_labels.append(pred_label)

    logits = torch.cat(logits, dim=0).detach().cpu().numpy()
    confidences = torch.cat(confidences, dim=0).detach().cpu().numpy()
    pred_labels = torch.cat(pred_labels, dim=0).detach().cpu().numpy()

    num_outliers = math.ceil(outlier_ratio * len(confidences))
    conf_sorted = np.argsort(confidences)
    pred_outlier = conf_sorted[-num_outliers:]
    pred_labels = np.zeros(len(confidences))
    pred_labels[pred_outlier] = 1

    return confidences, pred_labels


def get_threshold_inlier(sampling_score):
    """
    get the threshold for sample selection (inlier)
    """

    if len(sampling_score) > 1:
        threshold = max(torch.mean(sampling_score) - 1 * (torch.std(sampling_score)), min(sampling_score))
    else:
        threshold = sampling_score

    return threshold


def get_threshold_outlier(sampling_score):
    """
    get the threshold for sample selection (outlier)
    """
    if len(sampling_score) > 1:
        threshold = min(torch.mean(sampling_score) - 1 * (torch.std(sampling_score)), max(sampling_score))
    else:
        threshold = sampling_score

    return threshold


def initialize_center(args, pred_inlier_idx, data, model):
    """
    """
    select_clean_inlier = data[pred_inlier_idx, :]
    train_dataset = MyDataSetRecon(select_clean_inlier)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=None, pin_memory=False)
    n_samples = 0
    eps = 0.1
    c = torch.zeros(model.hidden_dim).cuda()

    model = model.cuda()
    model.train()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            data = data.cuda()
            outputs = model(data)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def sample_selection(args, loss, pred_entropy, label, data, idx, epoch):
    """

    """
    # select all data during warm up
    if epoch < args.epochs_warmup:
        clean_idx = torch.tensor(np.arange(len(idx))).cuda()
        non_select_idx = torch.tensor(np.arange(len(idx))).cuda()

    else:
        # compute gradient
        gradients = torch.autograd.grad(loss, data, grad_outputs=torch.ones_like(loss), allow_unused=True, retain_graph=True)[0]
        # compute the gradient norm
        gradients_norm = torch.clamp(torch.norm(gradients, dim=1, p=2), 1e-10, 0.4)
        # gradients_norm = (gradients_norm - torch.min(gradients_norm)) / (torch.max(gradients_norm) - torch.min(gradients_norm) + 1e-10)
        # loss = (loss - torch.min(loss)) / (torch.max(loss) - torch.min(loss) + 1e-10)  # normalize
        # pred_entropy = (pred_entropy - torch.min(pred_entropy)) / (torch.max(pred_entropy) - torch.min(pred_entropy) + 1e-10)  # normalize
        sampling_score = (args.alpha * loss) + ((1-args.alpha) * pred_entropy)
        # sampling_score = loss

        inlier_idx = torch.where(label == 0)[0]
        outlier_idx = torch.where(label == 1)[0]
        sampling_score_inlier = sampling_score[inlier_idx.cuda()]
        sampling_score_outlier = sampling_score[outlier_idx.cuda()]
        # print(sampling_score_outlier)

        if len(sampling_score_inlier) > 1:
            sampling_score_inlier = torch.clamp(sampling_score_inlier, .000000001, 10)
            power_transformer = PowerTransformer(method='box-cox')
            sampling_score_inlier = torch.from_numpy(power_transformer.fit_transform(sampling_score_inlier.detach().cpu().reshape(-1, 1)).astype(np.float32)).flatten().cuda()

        if len(sampling_score_outlier) > 1:
            sampling_score_outlier = torch.clamp(sampling_score_outlier, .000000001, 10)
            power_transformer = PowerTransformer(method='box-cox')
            sampling_score_outlier = torch.from_numpy(power_transformer.fit_transform(sampling_score_outlier.detach().cpu().reshape(-1, 1)).astype(np.float32)).flatten().cuda()

        # determine threshold for clean easy data
        threshold_clean_easy_inlier = get_threshold_inlier(sampling_score_inlier)
        threshold_clean_easy_outlier = get_threshold_outlier(sampling_score_outlier)

        clean_easy_idx_inlier = torch.nonzero(sampling_score_inlier <= threshold_clean_easy_inlier).flatten()
        clean_easy_idx_outlier = torch.nonzero(sampling_score_outlier <= threshold_clean_easy_outlier).flatten()
        clean_easy_idx = torch.concat((inlier_idx[clean_easy_idx_inlier], outlier_idx[clean_easy_idx_outlier]), dim=0)

        # select clean hard sample
        non_select_idx = torch.nonzero(~torch.isin(torch.arange(len(idx)).cuda(), clean_easy_idx)).flatten()
        non_select_idx_inlier = torch.tensor(np.intersect1d(inlier_idx.flatten().cpu(), non_select_idx.cpu())).cuda()
        non_select_idx_outlier = torch.tensor(np.intersect1d(outlier_idx.flatten().cpu(), non_select_idx.cpu())).cuda()
        sampling_score_hard = args.alpha * loss + (1 - args.alpha) * (1 - gradients_norm)
        # sampling_score_hard = args.alpha * loss + (1 - args.alpha) * (1 - pred_entropy)
        sampling_score_hard_inlier = sampling_score_hard[non_select_idx_inlier]
        sampling_score_hard_outlier = sampling_score_hard[non_select_idx_outlier]

        if len(sampling_score_hard_inlier) > 1:
            sampling_score_hard_inlier = torch.clamp(sampling_score_hard_inlier, .000000001, 10)
            power_transformer = PowerTransformer(method='box-cox')
            sampling_score_hard_inlier = torch.from_numpy(power_transformer.fit_transform(sampling_score_hard_inlier.detach().cpu().reshape(-1, 1)).astype(np.float32)).flatten().cuda()

        if len(sampling_score_hard_outlier) > 1:
            sampling_score_hard_outlier = torch.clamp(sampling_score_hard_outlier, .000000001, 10)
            power_transformer = PowerTransformer(method='box-cox')
            sampling_score_hard_outlier = torch.from_numpy(power_transformer.fit_transform(sampling_score_hard_outlier.detach().cpu().reshape(-1, 1)).astype(np.float32)).flatten().cuda()

        # determine threshold for clean hard data
        threshold_clean_hard_inlier = get_threshold_inlier(sampling_score_hard_inlier)
        threshold_clean_hard_outlier = get_threshold_outlier(sampling_score_hard_outlier)

        clean_hard_idx_inlier = torch.nonzero(sampling_score_hard_inlier <= threshold_clean_hard_inlier).flatten()
        clean_hard_idx_outlier = torch.nonzero(sampling_score_hard_outlier <= threshold_clean_hard_outlier).flatten()
        clean_hard_idx = torch.concat((non_select_idx_inlier[clean_hard_idx_inlier], non_select_idx_outlier[clean_hard_idx_outlier]), dim=0)

        clean_idx = torch.concat((clean_easy_idx, clean_hard_idx), dim=0)
        non_select_idx = torch.nonzero(~torch.isin(torch.arange(len(idx)).cuda(), clean_idx)).flatten()

        # only clean easy data
        # clean_idx = clean_easy_idx
        # non_select_idx = torch.nonzero(~torch.isin(torch.arange(len(idx)).cuda(), clean_idx)).flatten()

    return clean_idx, non_select_idx


def label_correction(args, dim, epoch, data, non_select_set, clean_set, noisy_label, ground_truth):
    """
    correct the label using selected clean data during the epoch
    and use the corrected label for further training

    dim: dimension of feature
    non_select_set: the set of data that is not selected (need to be label corrected)
    clean_set: the set of data that is selected (clean data)
    noisy_label: the noisy label of the data
    ground_truth: the ground truth label of the data
    """
    non_select_distances = []
    # select clean inliers and outliers
    clean_inlier_idx = torch.where(noisy_label[clean_set] == 0)[0]
    clean_outlier_idx = torch.where(noisy_label[clean_set] == 1)[0]
    clean_inlier_idx = clean_set[clean_inlier_idx]
    clean_outlier_idx = clean_set[clean_outlier_idx]

    non_select_inlier_idx = torch.nonzero((noisy_label[non_select_set] == 0)).flatten()
    non_select_outlier_idx = torch.nonzero((noisy_label[non_select_set] == 1)).flatten()

    # initialize the model and optimizer
    contrastive_model = Contrastive_Model(feature_dim = dim)
    contrastive_optimizer = torch.optim.Adam(contrastive_model.parameters(), lr=args.learning_rate_contrastive)
    # initialize the inliers center
    center = initialize_center(args = args, pred_inlier_idx = clean_inlier_idx, data = data, model = contrastive_model)
    # initialize the contrastive loss
    contrastive_loss = Contrastive_Loss(center = center)

    # train contrastive leaning model on clean data
    clean_data = data[clean_set, :-1]
    clean_label = noisy_label[clean_set]
    clean_data = np.concatenate((clean_data, clean_label), axis=1)
    train_dataset = MyDataSet(clean_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=None, pin_memory=False)
    non_select_data = data[non_select_set, :]
    eval_dataset = MyDataSet(non_select_data)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=None, pin_memory=False)

    contrastive_model.train()
    for epoch in range(10):
        for _, data, label in train_loader:
            data, label = data.cuda(), label.cuda()
            outputs = contrastive_model(data) # (batch size, hidden dim)
            # loss
            loss = contrastive_loss(outputs, label)
            # update model parameters
            contrastive_optimizer.zero_grad()
            loss.backward()
            contrastive_optimizer.step()

    # evaluate the non-select dataset
    contrastive_model.eval()
    for _, data, label in eval_loader:
        data = data.cuda()
        outputs = contrastive_model(data)
        non_select_distance = torch.norm(outputs - center.unsqueeze(0), p=2, dim = 1)  # (batch size, 1)
        non_select_distances.append(non_select_distance)

    non_select_distances = torch.cat(non_select_distances, dim=0).detach().cpu()

    # non_select_distances_inlier = non_select_distances[non_select_inlier_idx]
    # non_select_distances_outlier = non_select_distances[non_select_outlier_idx]
    # if len(non_select_distances_inlier) > 1:
    #     power_transformer = PowerTransformer(method='box-cox')
    #     non_select_distances_inlier = torch.from_numpy(power_transformer.fit_transform(non_select_distances_inlier.detach().cpu().reshape(-1, 1)).astype(np.float32)).flatten().cuda()
    #
    # if len(non_select_distances_outlier) > 1:
    #     power_transformer = PowerTransformer(method='box-cox')
    #     non_select_distances_outlier = torch.from_numpy(power_transformer.fit_transform(non_select_distances_outlier.detach().cpu().reshape(-1, 1)).astype(np.float32)).flatten().cuda()
    #
    # # threshold_correct_label_inlier = get_threshold_outlier(non_select_distances_inlier)
    # threshold_correct_label_outlier = get_threshold_inlier(non_select_distances_outlier)
    #
    # correct_label_inlier_idx = torch.nonzero(non_select_distances_inlier >= threshold_correct_label_inlier).flatten()
    # correct_label_outlier_idx = torch.nonzero(non_select_distances_outlier <= threshold_correct_label_outlier).flatten()
    # correct_label_inlier_idx = non_select_inlier_idx[correct_label_inlier_idx]
    # correct_label_outlier_idx = non_select_outlier_idx[correct_label_outlier_idx]

    num_non_select_inlier = len(non_select_inlier_idx)
    num_non_select_outlier = len(non_select_outlier_idx)
    # number of labels need to be modified
    num_correct_inlier = math.ceil(num_non_select_inlier * 0.01) #0.01 others #0.1 for Pima, pendigits/ 0.1 thursday
    num_correct_outlier = math.ceil(num_non_select_outlier * 0.2) #0.2
    # sorted idx of distance
    sorted_distances_idx = torch.argsort(non_select_distances)
    correct_label_inlier_idx = sorted_distances_idx[-num_correct_inlier:]
    correct_label_outlier_idx = sorted_distances_idx[:num_correct_outlier]

    # update the labels
    corrected_label = noisy_label.clone()
    corrected_label[non_select_set[correct_label_inlier_idx]] = 1
    corrected_label[non_select_set[correct_label_outlier_idx]] = 0

    print('f1 score of corrected label: ', f1_score(corrected_label, ground_truth))

    return corrected_label


def filter_outlier_pool(args, dim, idx, data, clean_set, non_select_set, noisy_label):
    """
    """
    model_filter = Reconstruct_Model(feature_dim = dim)
    model_filter = model_filter.cuda()
    optimizer = torch.optim.Adam(model_filter.parameters(), lr=args.learning_rate_reconstruct)
    loss_function = nn.MSELoss()
    loss_function_raw = nn.MSELoss(reduction='none')

    # select clean inliers data and outliers
    clean_inlier_idx = torch.where(noisy_label[clean_set] == 0)[0]
    clean_inlier_idx = clean_set[clean_inlier_idx]
    clean_inlier_data = data[clean_inlier_idx, :]
    select_outlier_idx = torch.where(noisy_label[clean_set] == 1)[0]
    select_outlier_idx = clean_set[select_outlier_idx]
    select_outlier_data = data[select_outlier_idx, :]

    train_inlier_dataset = MyDataSetRecon(clean_inlier_data)
    train_inlier_loader = DataLoader(train_inlier_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=None, pin_memory=False)
    train_outlier_dataset = MyDataSetRecon(select_outlier_data)
    train_outlier_loader = DataLoader(train_outlier_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=None, pin_memory=False)

    # train model on selected clean inliers
    model_filter.train()
    for i in range(10): # 40 epochs pageblock/10 epochs others
        for data in train_inlier_loader:
            data = data.cuda()
            outputs = model_filter(data)
            loss = loss_function(outputs, data)
            # update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # evaluate the model
    loss_outlier_pool = []
    model_filter.eval()
    for data in train_outlier_loader:
        data = data.cuda()
        outputs = model_filter(data)
        # print(outputs)
        loss = loss_function_raw(outputs, data).mean(dim=1)
        # print(loss)
        loss_outlier_pool.append(loss)

    loss_outlier_pool = torch.cat(loss_outlier_pool, dim=0).detach().cpu()
    loss_outlier_pool = (loss_outlier_pool - torch.min(loss_outlier_pool)) / (torch.max(loss_outlier_pool) - torch.min(loss_outlier_pool))  # normalize

    # # filter the selected outlier pool
    if len(loss_outlier_pool) > 1:
        loss_outlier_pool = torch.clamp(loss_outlier_pool, .000000001, 10)
        power_transformer = PowerTransformer(method='box-cox')
        loss_outlier_pool = torch.from_numpy(power_transformer.fit_transform(loss_outlier_pool.detach().cpu().reshape(-1, 1)).astype(np.float32)).flatten().cuda()

    # determine threshold for clean outlier
    threshold_clean_outlier = min(torch.mean(loss_outlier_pool) + 1 * (torch.std(loss_outlier_pool)), max(loss_outlier_pool))
    clean_outlier_idx = torch.nonzero(loss_outlier_pool >= threshold_clean_outlier).flatten().cpu()

    # loss_outlier_pool = torch.argsort(loss_outlier_pool)
    # num_outliers = math.ceil(len(loss_outlier_pool) * 0.1)
    # clean_outlier_idx = loss_outlier_pool[-num_outliers:]
    # clean_outlier_idx = select_outlier_idx[clean_outlier_idx]
    print('filtered clean outlier number', len(clean_outlier_idx))
    print('f1 score of filtered outlier ', f1_score(noisy_label[clean_outlier_idx].cpu().numpy(), ground_truth[clean_outlier_idx].cpu().numpy()))
    noisy_outlier_idx = select_outlier_idx[~torch.isin(select_outlier_idx, clean_outlier_idx)].flatten()

    return clean_inlier_idx, clean_outlier_idx, noisy_outlier_idx


def run_NN(args, dim, train_dataloader, eval_dataloader, noisy_label, soft_label, ground_truth, full_data):
    """

    """
    # 初始化模型
    model = Model(feature_dim=dim, num_classes=2)
    model = model.cuda()
    ema_recorder = EMARecorder(noisy_label)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = SoftLabelLoss() # use to collect batch average loss
    loss_function_raw = SoftLabelLoss(reduction='none') # collect per sample loss

    model.train()
    for epoch in range(args.epochs):

        clean_set = []
        non_select_set = []
        pred_probs = []
        pred_labels = []
        indices = []
        for idx, data, label in train_dataloader:
            data, label = data.cuda(), label.cuda()
            data.requires_grad = True
            pred_logits = model(data)
            pred_prob = torch.softmax(pred_logits, dim=1)

            # outlier ratio in current batch
            noisy_outlier_ratio = noisy_label[idx].float().mean().item()
            num_outliers = math.ceil(noisy_outlier_ratio * len(idx))
            outlier_score = torch.argsort(pred_prob[:, 1]) # outlier score
            pred_outlier = outlier_score[-num_outliers:]
            pred_label = torch.zeros(len(idx)).cuda().to(int)
            pred_label[pred_outlier] = 1
            pred_label = torch.argmax(pred_prob, dim=1)  # predicted label
            pred_probs.append(pred_prob)
            pred_labels.append(pred_label)
            indices.append(idx)

            # loss and entropy score
            pred_entropy = -torch.sum(pred_prob * torch.log(pred_prob + 1e-10), 1)
            loss = loss_function_raw(pred_logits, label, soft_label[idx])

            # 计算ema
            if epoch <= 0:  # dont calculate EMA for first epoch
                loss = loss
                pred_entropy = pred_entropy
            else:
                loss = (args.beta * ema_recorder.get_loss_ema(idx.cuda())) + ((1 - args.beta) * loss.cuda())
                pred_entropy = (args.beta * ema_recorder.get_entropy_ema(idx.cuda())) + ((1 - args.beta) * pred_entropy.cuda())

            ema_recorder.record_loss_ema(idx.cuda(), loss)  # record the ema loss for each sample
            ema_recorder.record_entropy_ema(idx.cuda(), pred_entropy)  # record ema pred entropy score for each sample

            # sample selection
            clean_idx, non_select_idx = sample_selection(args, loss, pred_entropy, label, data, idx, epoch)
            clean_set.append(idx.cuda()[clean_idx])
            non_select_set.append(idx.cuda()[non_select_idx])

            # update model parameters
            optimizer.zero_grad()
            train_loss = loss_function(pred_logits[clean_idx], label[clean_idx], soft_label[idx.cuda()[clean_idx]])
            train_loss.backward()
            optimizer.step()

        if epoch >= args.epochs_warmup:
            clean_set = torch.cat(clean_set, dim=0).cpu()
            non_select_set = torch.cat(non_select_set, dim=0).cpu()
            print(len(clean_set), len(torch.where(noisy_label[clean_set] == 0)[0]),len(torch.where(noisy_label[clean_set] == 1)[0]))

            # filter clean outlier pool
            # clean_inlier_idx, clean_outlier_idx, noisy_outlier_idx = filter_outlier_pool(args, dim, idx, full_data, clean_set, non_select_set, noisy_label)
            # clean_set = torch.concat((clean_inlier_idx, clean_outlier_idx), dim=0).cpu()
            # non_select_set = torch.concat((non_select_set, noisy_outlier_idx), dim=0).cpu()

            # correct noisy labels
            # Mammography/satellite/
            # if epoch % 5 == 0:
            #     noisy_label = label_correction(args=args, dim=dim, epoch = epoch, data = full_data, non_select_set=non_select_set, clean_set=clean_set, noisy_label=noisy_label, ground_truth=ground_truth)

            # correct labels every epoch
            # others
            noisy_label = label_correction(args=args, dim=dim, epoch = epoch, data = full_data, non_select_set=non_select_set, clean_set=clean_set, noisy_label=noisy_label, ground_truth=ground_truth)

        # update soft label
        if epoch >= args.epochs_warmup:
            indices = torch.cat(indices, dim=0)
            pred_probs = torch.cat(pred_probs, dim=0).detach().cpu()
            soft_label = pred_probs[torch.argsort(indices)]
            soft_label = soft_label.cuda()

        # inference model
        confidences, pred_labels = inference_NN(args = args, model = model, eval_loader = eval_dataloader, outlier_ratio= noisy_outlier_ratio)

        # update the dataloader
        train_clean_bad_set = np.concatenate((full_data[:,:-1], noisy_label.cpu().numpy().reshape(-1,1)), axis=1)
        train_dataset = MyDataSet(train_clean_bad_set)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=None, pin_memory=False)
        eval_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=None, pin_memory=False)

        # evaluate performance
        f1 = f1_score(ground_truth.numpy(), pred_labels)
        precision = precision_score(ground_truth.numpy(), pred_labels)
        recall = recall_score(ground_truth.numpy(), pred_labels)
        print(f'===Epoch {epoch}===')
        print(f"F1 score of model is {f1}")
        print(f"Precision score of model is {precision}")
        print(f"Recall score of model is {recall}")
        auc = roc_auc_score(ground_truth.numpy(), confidences)
        print(f"ROC AUC of model is {auc}")
        print(" ")

    return model


def mislabel_detector(args, train_dataloader, eval_dataloader, dim, noisy_label, ground_truth, soft_label, outlier_ratio, full_data):
    """

    """
    correct_num = 0
    detect_num = 0
    correct_inlier = 0
    correct_outlier = 0
    logits = []
    confidences = []
    pred_labels = []
    noisy_label_tmp = noisy_label.clone()
    true_detect_idx = []

    # run network
    model = run_NN(args = args, dim = dim, train_dataloader= train_dataloader, eval_dataloader=eval_dataloader, noisy_label = noisy_label, soft_label = soft_label, ground_truth = ground_truth, full_data = full_data)

    # mislabel detection
    print('=====Detecting mislabel data=====')
    model.eval()

    for idx, data, label in eval_dataloader:
        data, label = data.cuda(), label.cuda()
        pred_logits = model(data)
        pred_prob = torch.softmax(pred_logits, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1)
        logits.append(pred_logits)
        confidences.append(pred_prob[:, 1])
        pred_labels.append(pred_label)

    logits = torch.concat(logits, axis=0).detach().cpu().numpy()
    confidences = torch.concat(confidences, axis=0).detach().cpu().numpy()
    pred_labels = torch.cat(pred_labels, dim=0).detach().cpu().numpy()

    num_outliers = math.ceil(outlier_ratio * len(confidences))
    conf_sorted = np.argsort(confidences)
    pred_outlier = conf_sorted[-num_outliers:]
    pred_labels = np.zeros(len(confidences))
    pred_labels[pred_outlier] = 1

    potential_bad_idx = np.where(pred_labels != np.squeeze(noisy_label_tmp.numpy()))[0]
    for idx in potential_bad_idx:
        detect_num += 1
        if (len(ground_truth) - 1) >= idx >= len(train_clean_set):
            correct_num += 1
            true_detect_idx.append(idx)
            if ground_truth[idx] == 0:
                correct_outlier += 1
            else:
                correct_inlier += 1

    print('===Report Performance===')
    print('Total Outlier Detected: ', np.sum(pred_labels))
    print('identified mislabeled outlier ', correct_outlier, 'identified mislabeled inlier ', correct_inlier)
    precision = correct_num / detect_num
    recall = correct_num / len(train_bad_set)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    print(f'Detection f1 score: {f1}')
    print(f'Detection precision score: {precision}')
    print(f'Detection recall score: {recall}')

    ratio = 1
    sample_size = math.ceil(len(potential_bad_idx) * ratio)
    correct_idx = np.random.choice(potential_bad_idx, sample_size, replace=False)
    noisy_label_tmp[correct_idx] = 1 - noisy_label_tmp[correct_idx]

    # noisy_label_tmp[true_detect_idx] = 1 - noisy_label_tmp[true_detect_idx]
    cleaned_dataset = np.concatenate((full_data[:,:-1], noisy_label_tmp.cpu().numpy().reshape(-1,1)), axis=1)
    print(cleaned_dataset.shape)
    # transfer to csv format
    cleaned_dataset = pd.DataFrame(cleaned_dataset)
    cleaned_dataset.to_csv(f'./cleaned_data/{args.dataset}_cleaned.csv', index=False, header=False)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Pendigits', help='dataset to run')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epochs', type=int, default=30, help='training epoch on clean data')
    parser.add_argument('--epochs_warmup', default=5, type=int, help='number of epochs to run warmup for (no sample selection)')
    parser.add_argument('--batch_size', type=int, default=128, help='detect batch size') # thursday btach size 64
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for sample selection')
    parser.add_argument('--learning_rate_contrastive', type=float, default=0.01, help='learning rate for contrastive model')
    parser.add_argument('--beta', default=0.9, type=float, help='controls how much weight should be given to historical data vs newer data for EMA')
    parser.add_argument('--alpha', default=0.5, type=float, help='tradeoff between loss and similarity score or confident score')

    args = parser.parse_args()
    seed = args.seed
    # set random seed
    random.seed(seed)
    torch.manual_seed(seed)  # cpu seed
    torch.cuda.manual_seed(seed)  # gpu seed
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        print(f"Running on {torch.cuda.get_device_name(0)}")

    # run LOF detector on dataset to generate noisy labels
    X, y_real, y_noisy, soft_label, dim, outlier_ratio = run_detector(args)

    # build dataloader
    train_dataloader, eval_dataloader, train_clean_bad_set, train_clean_set, train_bad_set, ground_truth, noisy_label, soft_label = create_dataloader(args, X, y_real, y_noisy, soft_label)

    # transform labels into tensor
    ground_truth = torch.tensor(ground_truth, dtype=torch.long)
    noisy_label = torch.tensor(noisy_label, dtype=torch.long)
    soft_label = torch.tensor(soft_label, dtype=torch.float).cuda()

    # run mislabel detector
    mislabel_detector(args = args, train_dataloader = train_dataloader, eval_dataloader = eval_dataloader, dim = dim, noisy_label = noisy_label, \
                      ground_truth = ground_truth, soft_label = soft_label, outlier_ratio = outlier_ratio, full_data = train_clean_bad_set)

