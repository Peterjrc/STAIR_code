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
from tqdm import tqdm
import torch.nn.functional as F
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.preprocessing import Normalizer, RobustScaler, MinMaxScaler
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from utils import *
from model import Model, Contrastive_Model, Reconstruct_Model, Model_cleanlab
from run_detector import run_detector
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues



def coteaching(model_1, model_2, device, train_clean_bad_set, train_clean_set, train_bad_set):

    noisy_label_tmp = noisy_label.clone()
    model1 = model_1.to(device)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    loss_function1_raw = nn.CrossEntropyLoss(reduction='none').to(device)
    loss_function1 = nn.CrossEntropyLoss().to(device)

    model2 = model_2.to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    loss_function2_raw = nn.CrossEntropyLoss(reduction='none').to(device)
    loss_function2 = nn.CrossEntropyLoss().to(device)

    train_loader1 = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=128, shuffle=False)
    train_loader2 = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=128, shuffle=True)

    model1.train()
    model2.train()
    for epo in range(35):
        remember_rate = 0.9
        for data in train_loader1:
            # GPU加速
            idx, train_feature, train_label = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            pred_logits_model1 = model1(train_feature)
            pred_logits_model2 = model2(train_feature)

            loss_model1 = loss_function1_raw(pred_logits_model1, train_label)
            loss_model2 = loss_function2_raw(pred_logits_model2, train_label)

            num_sample_keep = int(len(idx) * remember_rate)
            select_idx_model1 = torch.argsort(loss_model1, descending=False)[:num_sample_keep]
            select_idx_model2 = torch.argsort(loss_model2, descending=False)[:num_sample_keep]

            co_loss_model1 = loss_function1(pred_logits_model1[select_idx_model2], train_label[select_idx_model2])
            optimizer1.zero_grad()
            co_loss_model1.backward()
            optimizer1.step()

            co_loss_model2 = loss_function2(pred_logits_model2[select_idx_model1], train_label[select_idx_model1])
            optimizer2.zero_grad()
            co_loss_model2.backward()
            optimizer2.step()

        detect_num = 0
        correct_num = 0
        pred_labels = []
        confidences = []
        print(len(train_clean_bad_set), len(train_clean_set))
        if epo % 5 == 0:
            print('=====Detecting mislabel data=====')
            model1.eval()
            with torch.no_grad():
                num = 0
                for data in test_loader:
                    idx, test_feature, test_label = data
                    test_feature = test_feature.to(device)
                    test_label = test_label.to(device)
                    test_label_predict = model1(test_feature)
                    pred_prob = torch.softmax(test_label_predict, dim=1)
                    confidences.append(pred_prob[:, 1])
                    pred_label = torch.argmax(pred_prob, dim=1)
                    pred_labels.append(pred_label)
                    test_label_predict = test_label_predict.to(device)

                pred_labels = torch.cat(pred_labels, dim=0).detach().cpu().numpy()

                confidences = torch.concat(confidences, axis=0).detach().cpu().numpy()
                num_outliers = math.ceil(outlier_ratio * len(confidences))
                conf_sorted = np.argsort(confidences)
                pred_outlier = conf_sorted[-num_outliers:]
                pred_labels = np.zeros(len(confidences))
                pred_labels[pred_outlier] = 1

                f1 = f1_score(ground_truth.numpy(), pred_labels)
                precision = precision_score(ground_truth.numpy(), pred_labels)
                recall = recall_score(ground_truth.numpy(), pred_labels)
                print(f"F1 score of model is {f1}")
                potential_bad_idx = np.where(pred_labels != np.squeeze(noisy_label_tmp.numpy()))[0]
                for idx in potential_bad_idx:
                    detect_num += 1
                    if (len(ground_truth) - 1) >= idx >= len(train_clean_set):
                        correct_num += 1

                precision = correct_num / detect_num
                recall = correct_num / len(train_bad_set)
                f1 = 2 * (precision * recall) / (precision + recall)
                print(correct_num)
                print("第{}轮: precision:{},recall:{},f1:{}".format(epo + 1, precision, recall, f1))



def coteaching_plus(model_1, model_2, device, train_clean_bad_set, train_clean_set, train_bad_set):

    noisy_label_tmp = noisy_label.clone()
    model1 = model_1.to(device)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    loss_function1 = nn.CrossEntropyLoss()
    loss_function1 = loss_function1.to(device)

    model2 = model_2.to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    loss_function2 = nn.CrossEntropyLoss()
    loss_function2 = loss_function2.to(device)

    train_loader1 = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=128, shuffle=False)
    train_loader2 = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=128, shuffle=True)

    model1.train()
    model2.train()
    for epo in range(35):
        early_loss1 = np.zeros(len(train_clean_bad_set), dtype=np.float64)
        for data in tqdm(train_loader1, leave=False):
            # GPU加速
            idx, train_feature, train_label = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer1.zero_grad()
            train_label_predict = model1(train_feature)

            # GPU加速
            train_label_predict = train_label_predict.to(device)
            pre_label2 = np.zeros(len(train_clean_bad_set), dtype=np.int64)
            # train_loss = loss_function1(train_label_predict, train_label)
            for i in range(len(idx)):
                early_loss1[idx[i].item()] = loss_function1(train_label_predict[i].view(1,-1), train_label[i].view(-1))

        early_loss2 = np.zeros(len(train_clean_bad_set), dtype=np.float64)
        for data in tqdm(train_loader2, leave=False):
            # GPU加速
            idx, train_feature, train_label = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer2.zero_grad()
            train_label_predict = model2(train_feature)

            # GPU加速
            train_label_predict = train_label_predict.to(device)
            for i in range(len(idx)):
                early_loss2[idx[i].item()] = loss_function2(train_label_predict[i].view(1,-1), train_label[i].view(-1))

        model1_preds = torch.argmax(model1(torch.tensor(train_clean_bad_set[:,:-1]).to(device).float()), dim=1)
        model2_preds = torch.argmax(model2(torch.tensor(train_clean_bad_set[:,:-1]).to(device).float()), dim=1)
        disagree_idx = torch.nonzero(model1_preds != model2_preds).detach().cpu().numpy().flatten()
        print(len(disagree_idx))
        early_loss1 = torch.tensor(early_loss1)
        early_loss2 = torch.tensor(early_loss2)
        model1_select_idx = torch.argsort(early_loss1[disagree_idx], descending=False)
        model2_select_idx = torch.argsort(early_loss2[disagree_idx], descending=False)

        print(len(model1_select_idx))
        print(len(model2_select_idx))

        update1 = []
        for index in range(int(0.0001 *len(model1_select_idx))):
            update1.append(train_clean_bad_set[model1_select_idx.detach().cpu().numpy()[index], :])

        update2 = []
        for index in range(int(0.0001 * len(model2_select_idx))):
            update2.append(train_clean_bad_set[model2_select_idx.detach().cpu().numpy()[index], :])

        if len(update1) == 0:
            update_1 = np.array(train_clean_bad_set)
        else:
            update_1 = np.array(update1)

        if len(update2) == 0:
            update_2 = np.array(train_clean_bad_set)
        else:
            update_2 = np.array(update2)

        train_loader_loss1 = DataLoader(dataset=MyDataSet(update_1), batch_size=128, shuffle=True)
        train_loader_loss2 = DataLoader(dataset=MyDataSet(update_2), batch_size=128, shuffle=True)

        print('======cross update model parameters=====')
        for data in tqdm(train_loader_loss2, leave = False):
            # GPU加速
            idx, train_feature, train_label = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer1.zero_grad()
            train_label_predict = model1(train_feature)

            # GPU加速
            train_label_predict = train_label_predict.to(device)
            train_loss = loss_function1(train_label_predict, train_label)
            train_loss.backward()
            optimizer1.step()

        for data in tqdm(train_loader_loss1, leave = False):
            # GPU加速
            idx, train_feature, train_label = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer2.zero_grad()
            train_label_predict = model2(train_feature)

            # GPU加速
            train_label_predict = train_label_predict.to(device)
            train_loss = loss_function2(train_label_predict, train_label)
            train_loss.backward()
            optimizer2.step()

        detect_num = 0
        correct_num = 0
        confidences = []
        pred_labels = []
        print(len(train_clean_set), len(train_clean_set))
        if epo % 5 == 0:
            print('=====Detecting mislabel data=====')
            model1.eval()
            with torch.no_grad():
                num = 0
                for data in test_loader:
                    idx, test_feature, test_label = data
                    test_feature = test_feature.to(device)
                    test_label = test_label.to(device)
                    test_label_predict = model1(test_feature)
                    pred_prob = torch.softmax(test_label_predict, dim=1)
                    pred_label = torch.argmax(pred_prob, dim=1)
                    pred_labels.append(pred_label)
                    confidences.append(pred_prob[:, 1])
                    test_label_predict = test_label_predict.to(device)

                pred_labels = torch.cat(pred_labels, dim=0).detach().cpu().numpy()

                confidences = torch.concat(confidences, axis=0).detach().cpu().numpy()
                num_outliers = math.ceil(outlier_ratio * len(confidences))
                conf_sorted = np.argsort(confidences)
                pred_outlier = conf_sorted[-num_outliers:]
                pred_labels = np.zeros(len(confidences))
                pred_labels[pred_outlier] = 1

                f1 = f1_score(ground_truth.numpy(), pred_labels)
                precision = precision_score(ground_truth.numpy(), pred_labels)
                recall = recall_score(ground_truth.numpy(), pred_labels)
                print(f"F1 score of model is {f1}")

                potential_bad_idx = np.where(pred_labels != np.squeeze(noisy_label_tmp.numpy()))[0]
                for idx in potential_bad_idx:
                    detect_num += 1
                    if (len(ground_truth) - 1) >= idx >= len(train_clean_set):
                        correct_num += 1

                precision = correct_num / detect_num
                recall = correct_num / len(train_bad_set)
                f1 = 2 * (precision * recall) / (precision + recall)
                print("第{}轮: precision:{},recall:{},f1:{}".format(epo + 1, precision, recall, f1))


def cleanLab(total_len, clean_len, data):

    features = []
    labels = []
    for i in data:
        features.append(i[:-1])
        labels.append(i[-1])
    features = np.array(features).astype(np.float32)
    labels = np.array(labels).astype(np.int64)

    from skorch import NeuralNetClassifier
    from sklearn.model_selection import cross_val_predict

    label_index = data.shape[1] - 1
    model_skorch = NeuralNetClassifier(Model_cleanlab, module__feature_dim=label_index, optimizer=torch.optim.Adam, lr= 0.01, max_epochs = 200)

    from cleanlab.filter import find_label_issues
    pred_probs = cross_val_predict(
        model_skorch,
        features,
        labels,
        cv=5,
        method="predict_proba",
    )
    predicted_labels = pred_probs.argmax(axis=1)
    ranked_label_issues = find_label_issues(
        labels,
        pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    print(f"Cleanlab found {len(ranked_label_issues)} label issues.")
    print(ranked_label_issues)

    detect_num = len(ranked_label_issues)
    correct_num = 0
    for i in ranked_label_issues:
        if (total_len - 1) >= i >= clean_len:
            correct_num += 1

    precision = correct_num / detect_num
    recall = correct_num / len(train_bad_set)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("correct_num{}, detect_num:{}".format(correct_num, detect_num))
    print("detect precision：{}".format(precision))
    print("detect recall：{}".format(recall))
    print("detect f1 score：{}".format(f1))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='satellite', help='dataset to run')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epochs', type=int, default=30, help='training epoch on clean data')
    parser.add_argument('--epochs_warmup', default=5, type=int,
                        help='number of epochs to run warmup for (no sample selection)')
    parser.add_argument('--batch_size', type=int, default=128, help='detect batch size')  # thursday btach size 64
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for sample selection')
    parser.add_argument('--learning_rate_reconstruct', type=float, default=0.01,
                        help='learning rate for reconstruct model')
    parser.add_argument('--learning_rate_contrastive', type=float, default=0.01,
                        help='learning rate for contrastive model')
    parser.add_argument('--beta', default=0.9, type=float,
                        help='controls how much weight should be given to historical data vs newer data for EMA')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='tradeoff between loss and similarity score or confident score')

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

    X, y_real, y_noisy, soft_label, dim, outlier_ratio = run_detector(args)

    # build dataloader
    train_dataloader, eval_dataloader, train_clean_bad_set, train_clean_set, train_bad_set, ground_truth, noisy_label, soft_label = create_dataloader(
        args, X, y_real, y_noisy, soft_label)

    # transform labels into tensor
    ground_truth = torch.tensor(ground_truth, dtype=torch.long)
    noisy_label = torch.tensor(noisy_label, dtype=torch.long)
    soft_label = torch.tensor(soft_label, dtype=torch.float).cuda()
    dim = train_clean_bad_set.shape[1] - 1
    model_1 = Model(feature_dim=dim)
    model_2 = Model(feature_dim=dim)
    device = torch.device("cuda")

    # coteaching(model_1, model_2, device, train_clean_bad_set, train_clean_set, train_bad_set)
    # cleanLab(total_len=len(ground_truth), clean_len=len(train_clean_set), data=train_clean_bad_set)
    coteaching_plus(model_1, model_2, device, train_clean_bad_set, train_clean_set, train_bad_set)