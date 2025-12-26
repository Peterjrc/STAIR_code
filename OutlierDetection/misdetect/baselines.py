from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import pandas
import torch
# from get_data import *
# from model import *
import operator
import math
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from collections import OrderedDict
# from pyod.models.knn import KNN


k = 320
k2 = 384

class Model(nn.Module):
    def __init__(self, input_len):
        super(Model, self).__init__()
        self.net1 = Sequential(OrderedDict([
            ('batch norm1', nn.BatchNorm1d(input_len, momentum=0.99)),
            ('linear1', nn.Linear(input_len, k)),
            ('relu1', nn.ReLU())
        ]))
        self.net2 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(k, momentum=0.99)),
            ('linear2', nn.Linear(k, k)),
            ('relu2', nn.ReLU())
        ]))
        self.net10 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(k, momentum=0.99)),
            ('linear2', nn.Linear(k, k2)),
            ('relu2', nn.ReLU())
        ]))
        self.net11 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(k2, momentum=0.99)),
            ('linear2', nn.Linear(k2, k2)),
            ('relu2', nn.ReLU())
        ]))
        self.net12 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(k2, momentum=0.99)),
            ('linear2', nn.Linear(k2, k)),
            ('relu2', nn.ReLU())
        ]))
        self.net13 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(k, momentum=0.99)),
            ('linear2', nn.Linear(k, k)),
            ('relu2', nn.ReLU())
        ]))
        self.net3 = Sequential(OrderedDict([
            ('batch norm3', nn.BatchNorm1d(k, momentum=0.99)),
            ('linear3', nn.Linear(k, 7)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        a = self.net1(x)
        b = self.net2(a) + a
        h1 = self.net10(b)
        h2 = self.net11(h1)
        h3 = self.net12(h2)
        h4 = self.net13(h3)
        c = self.net3(h4)
        c = c.squeeze(-1)
        return c



class MyDataSet(Dataset):
    def __init__(self, loaded_data):
        self.data = loaded_data
        self.labels = self.data[:, -1].astype(int)  # extract the hard labels
        self.features = torch.tensor(self.data[:, :-1], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feature = self.features[index]
        label = torch.tensor(self.labels[index], dtype=torch.int64)
        return index, feature, label


# GPU
device = torch.device("cuda")
est_label_predict.to(device)


def coteaching(model_1, model_2, device, train_clean_bad_set, train_clean_set, train_bad_set):

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
    for epo in range(200):
        remember_rate = 0.5
        for data in tqdm(train_loader1, leave=False):
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
        predict_labels = []
        print(len(train_clean_bad_set), len(train_clean_set))
        if epo % 10 == 0:
            print('=====Detecting mislabel data=====')
            model1.eval()
            with torch.no_grad():
                num = 0
                for data in test_loader:
                    idx, test_feature, test_label = data
                    test_feature = test_feature.to(device)
                    test_label = test_label.to(device)
                    test_label_predict = model1(test_feature)
                    test_label_predict = t.append(test_label_predict.argmax(1).cpu().numpy())
                    # if test_label_predict.argmax(1) != test_label:
                    #     detect_num += 1
                    #     if (len(train_clean_bad_set) - 1) >= num >= len(train_clean_set):
                    #         correct_num += 1
                    # num += 1
                predict_labels = np.concatenate(predict_labels, axis = 0)
                potential_bad_idx = np.where(predict_labels != train_clean_bad_set[:, -1])[0]
                for idx in potential_bad_idx:
                    detect_num += 1

                    if (len(train_clean_bad_set) - 1) >= idx >= len(train_clean_set):
                        correct_num += 1
                precision = correct_num / detect_num
                recall = correct_num / len(train_bad_set)
                f1 = 2 * (precision * recall) / (precision + recall)
                print("第{}轮: precision:{},recall:{},f1:{}".format(epo + 1, precision, recall, f1))



def coteaching_plus(model_1, model_2, device, train_clean_bad_set, train_clean_set, train_bad_set):
    # model1 = Model()
    model1 = model_1.to(device)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    loss_function1 = nn.CrossEntropyLoss()
    loss_function1 = loss_function1.to(device)

    # model2 = Model()
    model2 = model_2.to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    loss_function2 = nn.CrossEntropyLoss()
    loss_function2 = loss_function2.to(device)

    train_loader1 = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=1, shuffle=False)
    train_loader2 = DataLoader(dataset=MyDataSet(train_clean_bad_set), batch_size=128, shuffle=True)

    model1.train()
    model2.train()
    for epo in range(200):
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
        for index in range(int(0.5*len(model1_select_idx))):
            update1.append(train_clean_bad_set[model1_select_idx.detach().cpu().numpy()[index], :])

        update2 = []
        for index in range(int(0.5 * len(model2_select_idx))):
            update2.append(train_clean_bad_set[model2_select_idx.detach().cpu().numpy()[index], :])

        update_1 = np.array(update1)
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
        print(len(train_clean_set), len(train_clean_set))
        if epo % 10 == 0:
            print('=====Detecting mislabel data=====')
            model1.eval()
            with torch.no_grad():
                num = 0
                for data in test_loader:
                    idx, test_feature, test_label = data
                    test_feature = test_feature.to(device)
                    test_label = test_label.to(device)
                    test_label_predict = model1(test_feature)
                    test_label_predict = test_label_predict.to(device)
                    if test_label_predict.argmax(1) != test_label:
                        detect_num += 1
                        if (len(train_clean_bad_set) - 1) >= num >= len(train_clean_set):
                            correct_num += 1
                    num += 1

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
    model_skorch = NeuralNetClassifier(Model, module__input_len=label_index)

    from cleanlab.filter import find_label_issues
    pred_probs = cross_val_predict(
        model_skorch,
        features,
        labels,
        cv=10,
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
    recall = correct_num / len(data)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("correct_num{}, detect_num:{}".format(correct_num, detect_num))
    print("detect precision：{}".format(precision))
    print("detect recall：{}".format(recall))
    print("detect f1 score：{}".format(f1))



