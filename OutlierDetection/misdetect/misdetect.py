import torch
from torch.utils.data import Dataset
import pandas as pd
import copy
import torch.nn as nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from collections import OrderedDict
import numpy as np
import time
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import torch.nn.functional as F
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.preprocessing import Normalizer, RobustScaler, MinMaxScaler
from sklearn import metrics
import torch.nn.init as init
from EntropyEarlyStop import ModelEntropyEarlyStop, cal_entropy
from torch.utils.data import WeightedRandomSampler
from baselines import *


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def init_weights(m):
    """
    weights initialization for nn
    """
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)  # Xavier 初始化
        init.zeros_(m.bias)


class NegEntropy(object):
    """
    negative entropy penalty for over-confident prediction
    """
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


def train_clean_dataset_builder(df_real, df_lof):
    """
    build training set for mislabel detection

    input:
    df_real: dataframe for real outlier dataset with al true labels
    df_lof: dataframe for lof predicted outlier dataset with some mislabels

    return:
    train_clean_bad_set: the training set with clean and mislabeled samples (sequential)
    train_clean_set: the clean training set
    train_bad_set: the mislabeled training set
    ground_truth: the true labels for the training set
    """

    good_sample_ratio = np.sum(df_real['label'] == df_lof['label']) / len(df_real)
    print(f"the good sample ratio is {good_sample_ratio}")
    columns = df_lof.columns.tolist()
    classes = len(df_lof.iloc[:, -1].unique()) # number of classes
    label_index = len(columns) - 1
    predicted_labels = np.array(df_lof.iloc[:, label_index]).reshape(-1,1)
    scaler = RobustScaler()

    X = df_lof.iloc[:, :-1].values
    X_transformed = scaler.fit_transform(X)
    train_set = np.concatenate([X_transformed, predicted_labels], axis = 1)

    ground_truth = np.array(df_real.iloc[:,-1]).reshape(-1,1)
    good_idx_array = np.where(predicted_labels == ground_truth)[0]
    all_idx_array = np.array(range(len(df_lof)))
    bad_idx_array = np.setdiff1d(all_idx_array, good_idx_array)

    train_clean_set = train_set[good_idx_array, :]
    train_bad_set = train_set[bad_idx_array, :]
    train_clean_bad_set = np.concatenate((train_clean_set, train_bad_set), axis = 0)  # we put mislabeled set under clean set to later verify if detected instance is truly mislabeled
    print(train_clean_set.shape, train_bad_set.shape)


    ground_truth = df_real.iloc[:,-1].values
    ground_truth = np.concatenate((ground_truth[good_idx_array], ground_truth[bad_idx_array]), axis = 0)
    noisy_label = df_lof.iloc[:,-1].values
    print(ground_truth.shape)

    return train_clean_bad_set, train_clean_set, train_bad_set, ground_truth, noisy_label


def LabelSmoothingCrossEntropy(logits, labels, smooth_rate=0.1, wa=0, wb=1):
    confidence = 1. - smooth_rate
    logprobs = F.log_softmax(logits, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = confidence * nll_loss + smooth_rate * smooth_loss
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)

    return torch.sum(loss)/num_batch


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


k = 320

# set the random seed here
# torch.manual_seed(0) # cpu seed
# torch.cuda.manual_seed(0) # gpu seed
# np.random.seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# torch.manual_seed(64)
# torch.cuda.manual_seed(64)
# np.random.seed(64)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# torch.manual_seed(32)
# torch.cuda.manual_seed(32)
# np.random.seed(32)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# torch.manual_seed(16)
# torch.cuda.manual_seed(16)
# np.random.seed(16)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# torch.manual_seed(8)
# torch.cuda.manual_seed(8)
# np.random.seed(8)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# questions to answer
# 1. why use batchnorm  facilitate training
# 2. why design simple MLP
# 3. why apply softmax
# 4. why skip connection
# 5. need to use SGD optimizer
# 6. large small batch comparison

kk = 768
class Model_retrieval(nn.Module):
    def __init__(self, input_len):
        super(Model_retrieval, self).__init__()
        self.net1 = Sequential(OrderedDict([
            ('batch norm1', nn.BatchNorm1d(input_len, momentum=0.99)),
            ('linear1', nn.Linear(input_len, kk)),
            ('relu1', nn.ReLU())
        ]))
        self.net2 = Sequential(OrderedDict([
            ('batch norm2', nn.BatchNorm1d(kk, momentum=0.99)),
            ('linear2', nn.Linear(kk, kk)),
            ('relu2', nn.ReLU())
        ]))
        self.net3 = Sequential(OrderedDict([
            ('batch norm3', nn.BatchNorm1d(kk, momentum=0.99)),
            ('linear3', nn.Linear(kk, 1)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        a = self.net1(x)
        b = self.net2(a) + a
        c = self.net3(b)
        return c



# k = 320
# k2 = 384
#
# class Model_loss(nn.Module):
#     def __init__(self):
#         super(Model_loss, self).__init__()
#         self.net1 = Sequential(OrderedDict([
#             ('batch norm1', nn.BatchNorm1d(label_index, momentum=0.99)),
#             ('linear1', nn.Linear(label_index, k)),
#             ('relu1', nn.ReLU())
#         ]))
#         self.net2 = Sequential(OrderedDict([
#             ('batch norm2', nn.BatchNorm1d(k, momentum=0.99)),
#             ('linear2', nn.Linear(k, k)),
#             ('relu2', nn.ReLU())
#         ]))
#         self.net10 = Sequential(OrderedDict([
#             ('batch norm2', nn.BatchNorm1d(k, momentum=0.99)),
#             ('linear2', nn.Linear(k, k2)),
#             ('relu2', nn.ReLU())
#         ]))
#         self.net11 = Sequential(OrderedDict([
#             ('batch norm2', nn.BatchNorm1d(k2, momentum=0.99)),
#             ('linear2', nn.Linear(k2, k2)),
#             ('relu2', nn.ReLU())
#         ]))
#         self.net12 = Sequential(OrderedDict([
#             ('batch norm2', nn.BatchNorm1d(k2, momentum=0.99)),
#             ('linear2', nn.Linear(k2, k)),
#             ('relu2', nn.ReLU())
#         ]))
#         self.net13 = Sequential(OrderedDict([
#             ('batch norm2', nn.BatchNorm1d(k, momentum=0.99)),
#             ('linear2', nn.Linear(k, k)),
#             ('relu2', nn.ReLU())
#         ]))
#         self.net3 = Sequential(OrderedDict([
#             ('batch norm3', nn.BatchNorm1d(k, momentum=0.99)),
#             ('linear3', nn.Linear(k, classes)),
#             ('sigmoid', nn.Sigmoid())
#         ]))
#
#     def forward(self, x):
#         a = self.net1(x)
#         b = self.net2(a) + a
#         h1 = self.net10(b)
#         h2 = self.net11(h1)
#         h3 = self.net12(h2)
#         h4 = self.net13(h3)
#         c = self.net3(h4)
#         c = c.squeeze(-1)
#         return c

class Model_loss(nn.Module):
    def __init__(self):
        super(Model_loss, self).__init__()
        self.fc1 = nn.Linear(label_index, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 50)
        self.bn3 = nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(50, classes)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out


def weight_sampler(data, noisy_label):
    """use to solve the imbalanced class issue by assigning larger weight to rare class"""
    class_sample_count = np.array([len(np.where(noisy_label == t)[0]) for t in np.unique(noisy_label)])
    print(class_sample_count)
    weight = 1. / class_sample_count
    noisy_label = noisy_label.astype(int)
    samples_weight = np.array([weight[t] for t in noisy_label])
    samples_weight = torch.tensor(samples_weight, dtype=torch.float)
    # samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler



def inference_NN(model, test_data, device = device, return_label = False):
    """

    inference labels after training model

    input:
    model: the trained model
    test_data: the test dataset
    device: the device to run the model
    return_label: whether to return the predicted labels

    return:
    predict_logits: the predicted logits
    predicted_labels: the predicted labels
    """
    test_dataloader = DataLoader(MyDataSet(test_data), batch_size=128, shuffle=False)
    model.eval()
    predict_logits = []
    true_labels = []

    for data in test_dataloader:
        _, test_feature, test_label = data
        test_feature = test_feature.to(device)
        test_label = test_label.to(device)
        net_out = model(test_feature)
        predict_logits.append(net_out.data.cpu().numpy())
        true_labels.append(test_label.data.cpu().numpy())

    if return_label:
        predict_proba = np.concatenate([F.softmax(torch.tensor(logits), dim=1).numpy() for logits in predict_logits], axis = 0)
        confidences = predict_proba[:, 1]
        true_labels = np.concatenate(true_labels, axis=0)
        predicted_labels = np.argmax(predict_proba, axis=1)
        # noisy_outlier_ratio = np.mean(true_labels)
        # num_outliers = math.ceil(noisy_outlier_ratio * len(confidences))
        # conf_sorted = np.argsort(confidences)
        # pred_outlier = conf_sorted[-num_outliers:]
        # pred_labels = np.zeros(len(confidences))
        # pred_labels[pred_outlier] = 1
        return np.concatenate(predict_logits, axis=0), predicted_labels

    else:
         return np.concatenate(predict_logits, axis=0)



def calculate_ema_loss(loss_matrix, alpha=0.9):

    n, k = loss_matrix.shape  # 获取样本数和epoch数
    ema_loss_matrix = np.zeros_like(loss_matrix)  # 初始化 EMA 损失矩阵

    # 初始化 EMA，第一个 epoch 就是原始损失
    ema_loss_matrix[:, 0] = loss_matrix[:, 0]

    # 从第二个 epoch 开始计算 EMA
    for t in range(1, k):
        ema_loss_matrix[:, t] = alpha * loss_matrix[:, t] + (1 - alpha) * ema_loss_matrix[:, t - 1]

    return ema_loss_matrix


def run_twin_NN(model_1, model_2, epochs, new_train_clean_bad_set, noisy_rate, noisy_label, separate_inlier_outlier):
    """
    run coteaching-based early loss detection

    input
    model_1: the first model
    model_2: the second model
    epochs: number of epochs
    new_train_clean_bad_set: the training set with clean and mislabeled samples
    noisy_rate: the noisy rate in dataset
    noisy_rate_outlier: the noisy rate for outliers
    noisy_rate_inlier: the noisy rate for inliers
    separate_inlier_outlier: whether to separate inliers and outliers

    """
    model1 = model_1.to(device)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    loss_function1_raw = nn.CrossEntropyLoss(reduction = 'none').to(device)
    loss_function1 = nn.CrossEntropyLoss().to(device)

    model2 = model_2.to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    loss_function2_raw = nn.CrossEntropyLoss(reduction = 'none').to(device)
    loss_function2 = nn.CrossEntropyLoss().to(device)

    train_loader1 = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=128, shuffle=False)
    train_loader2 = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=128, shuffle=True)

    loss2 = np.zeros((len(new_train_clean_bad_set), epochs), dtype=np.float64)
    loss1 = np.zeros((len(new_train_clean_bad_set), epochs), dtype=np.float64)

    model1.train()
    model2.train()
    for epo in range(epochs):

        for data in tqdm(train_loader1, leave=False):
            # GPU加速
            idx, train_feature, train_label = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            pred_logits_model1 = model1(train_feature)
            pred_logits_model2 = model2(train_feature)


            if separate_inlier_outlier:

                # calculate the outlier rate in current batch
                noisy_outlier_ratio = train_label.float().mean()
                threshold_outlier = 1 - noisy_outlier_ratio * min((epo / epochs), 1)
                noisy_inlier_ratio = 1 - noisy_outlier_ratio
                threshold_inlier = 1 - noisy_inlier_ratio * min((epo / epochs), 1)

                loss_model1 = loss_function1_raw(pred_logits_model1, train_label)
                loss_model2 = loss_function2_raw(pred_logits_model2, train_label)

                num_outliers = math.ceil(noisy_outlier_ratio * len(idx))
                model1_pred_prob = torch.softmax(pred_logits_model1, dim=1)
                model2_pred_prob = torch.softmax(pred_logits_model2, dim=1)
                model1_outlier_scores = model1_pred_prob[:, 1]
                model2_outlier_scores = model2_pred_prob[:, 1]

                conf_sorted_model1 = torch.argsort(model1_outlier_scores)
                conf_sorted_model2 = torch.argsort(model2_outlier_scores)
                pred_outlier_1 = conf_sorted_model1[-num_outliers:]
                pred_outlier_2 = conf_sorted_model2[-num_outliers:]

                model1_pred_label = torch.zeros(len(idx)).to(device).to(int)
                model1_pred_label[pred_outlier_1] = 1
                model2_pred_label = torch.zeros(len(idx)).to(device).to(int)
                model2_pred_label[pred_outlier_2] = 1

                # disagree_idx = torch.flatten(torch.nonzero(model1_pred_label != model2_pred_label))
                # print('length of disagree idx', disagree_idx)
                #
                # agree_idx = torch.flatten(torch.nonzero(model1_pred_label == model2_pred_label))
                # print('length of agree idx', agree_idx)

                model1_inlier_idx = torch.nonzero(model1_pred_label == 0).flatten().cpu().to(device)
                model1_outlier_idx = torch.nonzero(model1_pred_label == 1).flatten().cpu().to(device)
                model2_inlier_idx = torch.nonzero(model2_pred_label == 0).flatten().cpu().to(device)
                model2_outlier_idx = torch.nonzero(model2_pred_label == 1).flatten().cpu().to(device)

                num_sample_inliers_1 = int(len(model1_inlier_idx) * threshold_inlier)
                num_sample_inliers_2 = int(len(model2_inlier_idx) * threshold_inlier)
                num_sample_outliers_1 = int(len(model1_outlier_idx) * threshold_outlier)
                num_sample_outliers_2 = int(len(model2_outlier_idx) * threshold_outlier)

                loss_model1_inlier = loss_model1[model1_inlier_idx]
                loss_model1_outlier = loss_model1[model1_outlier_idx]
                loss_model2_inlier = loss_model2[model2_inlier_idx]
                loss_model2_outlier = loss_model2[model2_outlier_idx]

                # select small loss samples as clean
                selected_model1_inlier = torch.argsort(loss_model1_inlier, descending=False)[:num_sample_inliers_1]
                selected_model1_outlier = torch.argsort(loss_model1_outlier, descending=False)[:num_sample_outliers_1]
                selected_model2_inlier = torch.argsort(loss_model2_inlier, descending=False)[:num_sample_inliers_2]
                selected_model2_outlier = torch.argsort(loss_model2_outlier, descending=False)[:num_sample_outliers_2]

                model1_select_idx = torch.cat((model1_inlier_idx[selected_model1_inlier], model1_outlier_idx[selected_model1_outlier]))
                model2_select_idx = torch.cat((model2_inlier_idx[selected_model2_inlier], model2_outlier_idx[selected_model2_outlier]))

            else:

                remember_rate = 1 - noisy_rate * min((epo / epochs), 1)

                loss_model1 = loss_function1_raw(pred_logits_model1, train_label)
                loss_model2 = loss_function2_raw(pred_logits_model2, train_label)
                # confident penalty
                conf_penalty_1 = NegEntropy()
                penalty_1 = conf_penalty_1(pred_logits_model1)
                loss_model1 = loss_model1 + penalty_1

                conf_penalty_2 = NegEntropy()
                # confident penalty
                penalty_2 = conf_penalty_2(pred_logits_model2)
                loss_model2 = loss_model2 + penalty_2

                # number of samples to keep
                num_sample_keep = int(len(idx) * remember_rate)
                model1_select_idx = torch.argsort(loss_model1, descending=False)[:num_sample_keep]
                model2_select_idx = torch.argsort(loss_model2, descending=False)[:num_sample_keep]

                _, pred1 = torch.max(pred_logits_model1, dim = -1)
                _, pred2 = torch.max(pred_logits_model2, dim = -1)

                disagree_idx = torch.where(pred1 != pred2)[0]
                print('the length of disagree index is', len(disagree_idx))

            co_loss_model1 = loss_function1(pred_logits_model1[model2_select_idx], train_label[model2_select_idx])
            optimizer1.zero_grad()
            co_loss_model1.backward()
            optimizer1.step()

            co_loss_model2 = loss_function2(pred_logits_model2[model1_select_idx], train_label[model1_select_idx])
            optimizer2.zero_grad()
            co_loss_model2.backward()
            optimizer2.step()

        model1.eval()
        with torch.no_grad():
            num = 0
            for data in test_loader:
                idx, test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model1(test_feature)
                test_label_predict = test_label_predict.to(device)
                loss = loss_function1_raw(test_label_predict, test_label).detach().cpu().numpy()
                loss1[idx, epo] = loss
                num += 1

        model2.eval()
        with torch.no_grad():
            num = 0
            for data in test_loader:
                idx, test_feature, test_label = data
                test_feature = test_feature.to(device)
                test_label = test_label.to(device)
                test_label_predict = model2(test_feature)
                test_label_predict = test_label_predict.to(device)
                loss = loss_function2_raw(test_label_predict, test_label).detach().cpu().numpy()
                loss2[idx, epo] = loss
                num += 1

    ema_loss1 = calculate_ema_loss(loss1)[:,-1]
    ema_loss2 = calculate_ema_loss(loss2)[:,-1]

    pred_logits1, pred_label1 = inference_NN(model1, new_train_clean_bad_set, device = device, return_label=True)
    pred_logits2, pred_label2 = inference_NN(model2, new_train_clean_bad_set, device = device, return_label=True)

    pred_logits1 = torch.tensor(pred_logits1)
    pred_logits2 = torch.tensor(pred_logits2)
    m = (pred_logits1.softmax(1) + pred_logits2.softmax(1)) / 2
    kl_main = F.kl_div(pred_logits1.log_softmax(1), m, reduction='none')
    kl_weight = F.kl_div(pred_logits2.log_softmax(1), m, reduction='none')
    similarity_score = (0.5 * torch.sum(kl_main, 1) + 0.5 * torch.sum(kl_weight, 1)).cpu().numpy()

    # prob of sample being an outlier
    # prob_model1 = pred_logits1.softmax(1)[:, 1].detach().cpu().numpy()
    # prob_model2 = pred_logits2.softmax(1)[:, 1].detach().cpu().numpy()

    return ema_loss1, ema_loss2, similarity_score, model1, model2



def run_NN(data, data_length, epoch = 3, return_loss=False, converge = False, entropyStop = False,
           n_eval=100,k =100,R_down=0.1, weight_sample = False, correct_label=False, run_twin_net = False,
           noisy_rate = 0, noisy_label = None, separate_inlier_outlier = False):

    if converge: # train model until converge use larger batch
        epoch = args.train_epoch
        entropyStop = args.earlyStop # whether to use entropy early stop
        learning_rate, converge_batch_size =args.lr_converge, args.converge_batch_size
        k, R_down, n_eval = args.converge_k, args.Rdown, args.n_eval

        # whether do entropy based early stop
        if entropyStop:
            ES = ModelEntropyEarlyStop(k=k, R_down=R_down)
            N_eval = min(n_eval, data.shape[0])
            np.random.seed(64)
            eval_index = np.random.choice(data.shape[0], N_eval, replace=False)  # random sample evaluation set
            data_eval = data[eval_index]
            isStop = False
            train_loader = DataLoader(dataset=MyDataSet(data), batch_size=converge_batch_size, shuffle=True)
            eval_loader = DataLoader(dataset=MyDataSet(data_eval), batch_size=1, shuffle=False)

        else:
            train_loader = DataLoader(dataset=MyDataSet(data), batch_size=converge_batch_size, shuffle=True)
            test_loader = DataLoader(dataset=MyDataSet(data), batch_size=1, shuffle=False)


    else: # run early loss detection
        epoch = args.detect_epoch
        learning_rate, detect_batch_size = args.lr_detect, args.detect_batch_size
        # k, R_down, n_eval = args.early_detect_k, args.Rdown, args.n_eval

        train_loader = DataLoader(dataset=MyDataSet(data), batch_size=detect_batch_size, shuffle=True)
        test_loader = DataLoader(dataset=MyDataSet(data), batch_size=1, shuffle=False)

    model = Model_loss() # use MLP model
    model = model.to(device)
    # model.apply(he_init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    # loss_function = loss_function.to(device)

    model.train()
    early_loss = np.zeros((data_length, epoch), dtype=np.float64)
    predict_proba = np.zeros((data_length, epoch), dtype=np.float64)
    # early_loss_per_label = [[[], [], [], [], []] for _ in range(total_len)]  # store recent few epochs loss

    if run_twin_net: # run twin-net early loss detection

        # initialize two models with different random seed
        torch.manual_seed(0)
        model_1 = Model_loss()
        model_1.apply(init_weights)

        torch.manual_seed(42)
        model_2 = Model_loss()
        model_2.apply(init_weights)

        epochs = args.twin_net_epoch
        loss_list1, loss_list2, similarity_score, model1, model2  = run_twin_NN(model_1, model_2, epochs, data,
                                                                                noisy_rate, noisy_label, separate_inlier_outlier)

    else:
        for epo in range(epoch):
            for data in tqdm(train_loader, leave=True, dynamic_ncols=True):
                # GPU加速
                _, train_feature, train_label = data
                train_feature = train_feature.to(device)
                train_label = train_label.to(device)
                # print(train_label)
                optimizer.zero_grad()
                train_label_predict = model(train_feature)
                # recon_score = model(train_feature)
                # GPU加速
                train_label_predict = train_label_predict.to(device)
                # train_loss = LabelSmoothingCrossEntropy(epoch, train_label_predict, train_label)
                train_loss = loss_function(train_label_predict, train_label)
                # train_loss = torch.mean(recon_score) # batch average MSE loss
                train_loss.backward()
                optimizer.step()

                if entropyStop:
                    isStop = ES.step(eval_loader, model, n_eval) # iteratively
                    if isStop:
                        break
                if entropyStop and isStop:
                        break

            if return_loss: # collect early loss from each epoch

                model.eval()
                with torch.no_grad():
                    num = 0
                    correct_num = 0
                    for data in test_loader:
                        _, test_feature, test_label = data
                        test_feature = test_feature.to(device)
                        test_label = test_label.to(device)
                        test_label_predict = model(test_feature)
                        loss = loss_function(test_label_predict, test_label)
                        early_loss[num, epo] = loss
                        predict_proba[num, epo] = test_label_predict[0][test_label[0]]
                        if torch.argmax(test_label_predict[0]) == torch.argmax(test_label):
                            correct_num += 1

                        # # early_loss_per_label[num][epo].append(loss_per_label) # store recent few epochs loss
                        num += 1

    if return_loss: # do early loss detection (only specify to true if using single net early loss detection)
        if entropyStop: # if use entropy stop in early loss detetction
            return isStop, early_loss, predict_proba
        else:
            return None, early_loss, predict_proba

    if run_twin_net: # do twin net early loss detection
        return loss_list1, loss_list2, similarity_score, model1, model2

    else: # train on clean data phase
        if entropyStop: # return the best model from enrtopstopping
            model_best = ES.getBestModel()  # load the best model
            return model_best
        else: # return model after finishing all epochs
            return model



# 测试influence detect的效果
def misdetect(full_data, clean_data, bad_data, ground_truth, detect_num, detect_iterate,
              clean_data_len, bad_data_len, early_stop = True, separate_inlier_outlier = False, weight_sampler = False, run_twin_net = False, noisy_rate = 0):
    """
    full_data: full sample with noisy labels
    clean_data: sample with clean labels
    bad_data: sample with bad labels
    ground_truth: full sample ground truth (all clean labels)
    detect_num: number of detected instances in each iteration
    early_stop: whether to stop early if no improvement
    """
    total_len = len(full_data)
    clean_len = len(clean_data)
    bad_len = len(bad_data)
    detected_bad_labels = []
    new_train_clean_bad_set = copy.deepcopy(full_data) # used for the first iteration
    noisy_label = np.squeeze(new_train_clean_bad_set[:,-1]).astype(np.int64) # used to correct noisy label
    print('There are in total annotated outliers in noisy dataset:', int(np.sum(noisy_label)))
    train_clean_bad_set_copy = copy.deepcopy(full_data) # used to construct new_train_clean_bad_set for next iteration
    instance2idx = {tuple(j):i for i, j in enumerate(full_data[:,:-1])} # used to match instance to idx
    bad_data_full = bad_data

    # 总共的correct个数
    total_detect_num = 0
    total_correct_num = 0
    true_bad_detected_idx = []
    retrieval_pool = []
    clean_idx = []

    for times in range(detect_iterate):

        # total_len = len(new_train_clean_bad_set)
        inliers_idx = np.where(noisy_label == 0)[0]
        print('inlier index',inliers_idx)
        outliers_idx = np.where(noisy_label == 1)[0]
        print('outlier index',outliers_idx)

        print('---running early loss detection---')
        print(f'---in the whole data, outlier/inlier %: {len(outliers_idx)/len(full_data)}---')

        if run_twin_net: # run twin net early loss detection
            loss_list1, loss_list2, similarity_score, model1, model2 = run_NN(new_train_clean_bad_set, data_length=len(new_train_clean_bad_set), epoch=args.detect_epoch, return_loss=False,
                                                  converge=False, entropyStop=False, weight_sample=False, correct_label=False, run_twin_net=run_twin_net, noisy_rate=noisy_rate,
                                                  noisy_label=noisy_label, separate_inlier_outlier=True)

        else: # run single net early loss detection
            isStop, early_loss, predict_proba = run_NN(new_train_clean_bad_set, data_length=len(new_train_clean_bad_set), epoch=args.detect_epoch, return_loss=True,
                                                  converge=False, entropyStop=False, weight_sample=False, correct_label=False, run_twin_net=False, noisy_rate=noisy_rate,
                                                    noisy_label = noisy_label, separate_inlier_outlier=False)

        ##################################################################################
        # average loss based on window size of 3
        if run_twin_net:

            loss1 = (loss_list1 - np.min(loss_list1)) / (np.max(loss_list1) - np.min(loss_list1))  # normalize
            loss2 = (loss_list2 - np.min(loss_list2)) / (np.max(loss_list2) - np.min(loss_list2))

            history_loss = loss_list1 + loss_list2 # combine the history ema loss from two nets
            history_loss = (history_loss - np.min(history_loss)) / (np.max(history_loss) - np.min(history_loss))  # normalized history ema loss
            similarity_score = (similarity_score - np.min(similarity_score)) / (np.max(similarity_score) - np.min(similarity_score)) # normalize similarity score

            similarity_score_inlier = similarity_score[inliers_idx]
            similarity_score_outlier = similarity_score[outliers_idx]
            print('similarity score outlier mean', np.mean(similarity_score_outlier))
            print('similarity score outlier std', np.std(similarity_score_outlier))
            print('similarity score inlier mean', np.mean(similarity_score_inlier))
            print('similarity score inlier std', np.std(similarity_score_inlier))

            # sampling_score =  (0.5 * (1-similarity_score)) + ((1-0.5) * history_loss) # combine similarity score and history loss
            sampling_score = history_loss
            sampling_score_inlier = sampling_score[inliers_idx]
            sampling_score_outlier = sampling_score[outliers_idx]

            print('sampling score outlier', sampling_score_outlier)
            sampling_score_inlier = torch.clamp(torch.tensor(sampling_score_inlier), 0.0000001, 10) # clamp all sampling scores to positive number
            print(len(sampling_score_inlier))
            sampling_score_inlier_transformed = PowerTransformer(method='box-cox').fit_transform(np.array(sampling_score_inlier.detach().numpy()).reshape(-1, 1))

            num_trim = math.ceil(sampling_score_inlier_transformed.size * .05)
            indices_smallest = np.argsort(sampling_score_inlier_transformed, axis=0)[:num_trim]
            indices_largest = np.argsort(sampling_score_inlier_transformed, axis=0)[-num_trim:]
            drop = np.concatenate([indices_smallest, indices_largest])
            mask = np.isin(np.arange(sampling_score_inlier_transformed.shape[0]), drop, invert=True)
            sampling_score_trimmed_inliers = sampling_score_inlier_transformed[mask]
            print(len(sampling_score_inlier_transformed))
            sampling_score_outlier = torch.clamp(torch.tensor(sampling_score_outlier), 0.0000001, 10)
            sampling_score_outlier_transformed = PowerTransformer(method='box-cox').fit_transform(np.array(sampling_score_outlier.detach().numpy()).reshape(-1, 1))

            num_trim = math.ceil(sampling_score_outlier_transformed.size * .05)
            indices_smallest = np.argsort(sampling_score_outlier_transformed, axis=0)[:num_trim]
            indices_largest = np.argsort(sampling_score_outlier_transformed, axis=0)[-num_trim:]
            drop = np.concatenate([indices_smallest, indices_largest])
            mask = np.isin(np.arange(sampling_score_outlier_transformed.shape[0]), drop, invert=True)
            sampling_score_trimmed_outliers = sampling_score_outlier_transformed[mask]

            # calculate the threshold for inlier and outlier metrics
            threshold_clean_inlier = np.mean(sampling_score_trimmed_inliers) + 1 * np.std(sampling_score_trimmed_inliers)
            threshold_bad_inlier = np.mean(sampling_score_trimmed_inliers) - 1.5 * np.std(sampling_score_trimmed_inliers)

            threshold_clean_outlier = np.mean(sampling_score_trimmed_outliers) + 1 *np.std(sampling_score_trimmed_outliers)
            threshold_bad_outlier = np.mean(sampling_score_trimmed_outliers) - 1.5 *np.std(sampling_score_trimmed_outliers)

            print(len(sampling_score_outlier_transformed))
            # print( np.mean(history_loss_transformed), np.std(history_loss_transformed))
            print('index of bad outlier 1', np.where(sampling_score_outlier_transformed <= threshold_bad_outlier)[0])
            bad_idx_outlier = outliers_idx[np.where(sampling_score_outlier_transformed <= threshold_bad_outlier)[0]]
            print('index of bad outlier real', outliers_idx[np.where(sampling_score_outlier_transformed <= threshold_bad_outlier)[0]])
            clean_idx_outlier = outliers_idx[np.where(sampling_score_outlier_transformed >= threshold_clean_outlier)[0]]
            bad_idx_inlier = inliers_idx[np.where(sampling_score_inlier_transformed <= threshold_bad_inlier)[0]]
            clean_idx_inlier = inliers_idx[np.where(sampling_score_inlier_transformed >= threshold_clean_inlier)[0]]

            print(f'length of bad outlier index: {len(bad_idx_outlier)}')
            print(f'length of clean outlier index: {len(clean_idx_outlier)}')


            print(f'length of bad inlier index: {len(bad_idx_inlier)}')
            print(f'length of clean inlier index: {len(clean_idx_inlier)}')

            bad_data_idx = np.union1d(bad_idx_outlier, bad_idx_inlier)
            print(f'length of bad data index: {len(bad_data_idx)}')
            bad_data = new_train_clean_bad_set[bad_data_idx, :]

            clean_data_idx = np.union1d(clean_idx_inlier, clean_idx_outlier)
            print(f'length of clean data index: {len(clean_data_idx)}')
            clean_data = new_train_clean_bad_set[clean_data_idx, :]

        ##################################################################################
        else: # do single net early loss detection
            scaler = MinMaxScaler(feature_range=(0, 1))
            early_loss_sum = sorted(enumerate(np.squeeze(scaler.fit_transform(np.mean(early_loss, axis = 1).reshape(-1, 1) * 10))), key=lambda x: -x[1])
            early_loss_avg = np.squeeze(np.mean(early_loss, axis = 1)).reshape(-1,1)
            early_loss_std = np.squeeze(np.std(early_loss, axis = 1)).reshape(-1,1)
            predict_proba_avg = np.squeeze(np.mean(predict_proba, axis = 1)).reshape(-1,1)

            # scale the early loss and std

            early_loss_avg_norm = scaler.fit_transform(early_loss_avg)
            early_loss_std_norm = scaler.fit_transform(early_loss_std)

            # loss_list = list(enumerate(early_loss_avg_norm))
            loss_list = list(enumerate(early_loss_avg_norm))
            loss_values = np.squeeze(np.array([x[1] for x in loss_list]))
            loss_idx = np.array([x[0] for x in loss_list])
        ##################################################################################
            if separate_inlier_outlier:
                tmp_inlier_loss_idx = loss_idx[inliers_idx]
                tmp_inlier_loss_values = loss_values[tmp_inlier_loss_idx]

                tmp_inlier_loss_clean_list = sorted(list(zip(tmp_inlier_loss_idx, tmp_inlier_loss_values)), key=lambda x: x[1])
                tmp_inlier_loss_bad_list = sorted(list(zip(tmp_inlier_loss_idx, tmp_inlier_loss_values)), key=lambda x: -x[1])
                tmp_inlier_loss_clean_idx = [x[0] for x in tmp_inlier_loss_clean_list]
                tmp_inlier_loss_bad_idx = [x[0] for x in tmp_inlier_loss_bad_list]

                tmp_outlier_loss_idx = loss_idx[outliers_idx]
                tmp_outlier_loss_values = loss_values[tmp_outlier_loss_idx]
                tmp_outlier_loss_clean_list = sorted(list(zip(tmp_outlier_loss_idx, tmp_outlier_loss_values)), key=lambda x: x[1])
                tmp_outlier_loss_bad_list = sorted(list(zip(tmp_outlier_loss_idx, tmp_outlier_loss_values)), key=lambda x: -x[1])
                tmp_outlier_loss_clean_idx = [x[0] for x in tmp_outlier_loss_clean_list]
                tmp_outlier_loss_bad_idx = [x[0] for x in tmp_outlier_loss_bad_list]

                tmp_outlier_clean_data = new_train_clean_bad_set[tmp_outlier_loss_clean_idx, :] # fancy indexing
                outlier_clean_true_idx = list(set([instance2idx[tuple(i)] for i in tmp_outlier_clean_data[:, :-1]]))

                tmp_inlier_clean_data = new_train_clean_bad_set[tmp_inlier_loss_clean_idx, :] # fancy indexing
                inlier_clean_true_idx = list(set([instance2idx[tuple(i)] for i in tmp_inlier_clean_data[:, :-1]]))
                # tmp_clean_data_idx = loss_clean_idx[:clean_data_len]
                # # print(len(clean_data_idx))
                # tmp_clean_data = new_train_clean_bad_set[tmp_clean_data_idx, :]
                # tmp_clean_data_true_idx = list(set([instance2idx[tuple(i)] for i in tmp_clean_data[:,:-1]])) # corresponding clean data idx in index in full data

                num_add_inlier = 0
                idx = 0
                print(f'---retriving top {int(clean_data_len/2)} inlier clean data---')
                while num_add_inlier < int(clean_data_len / 2):
                    if inlier_clean_true_idx[idx] not in clean_idx:
                        clean_idx.append(inlier_clean_true_idx[idx])
                        num_add_inlier += 1
                        tmp = full_data[inlier_clean_true_idx[idx], :]
                        tmp_clean = np.append(tmp, 0) # 0 indicates clean data
                        retrieval_pool.append(tmp_clean)
                    idx += 1

                num_add_outlier = 0
                idx = 0
                print(f'---retriving top {int(clean_data_len/5)} outlier clean data---')
                while num_add_outlier < int(clean_data_len / 5):
                    if idx >= len(outlier_clean_true_idx):
                        break
                    if outlier_clean_true_idx[idx] not in clean_idx:
                        clean_idx.append(outlier_clean_true_idx[idx])
                        num_add_outlier += 1
                        tmp = full_data[outlier_clean_true_idx[idx], :]
                        tmp_clean = np.append(tmp, 0)
                        retrieval_pool.append(tmp_clean)
                    idx += 1
                ########################################################################################

                # 画 early loss图
                # if times == 0:
                #     loss_df = np.zeros((total_len, 2))
                #     loss_df[:, 0] = loss_values
                #     mislabel_inlier_idx = inlier_loss_idx[np.where(inlier_loss_idx > clean_len)[0]]
                #     clean_inlier_idx = inlier_loss_idx[np.where(inlier_loss_idx <= clean_len)[0]]
                #     mislabel_outlier_idx = outlier_loss_idx[np.where(outlier_loss_idx > clean_len)[0]]
                #     clean_outlier_idx = outlier_loss_idx[np.where(outlier_loss_idx <= clean_len)[0]]
                #     print('mislabel inlier', mislabel_inlier_idx)
                #
                #     print('mislabel outlier', mislabel_outlier_idx)
                #     # assert len(mislabel_inlier_idx) + len(mislabel_outlier_idx) == mislabel_num
                #
                #     loss_df[mislabel_inlier_idx, 1] = 1
                #     loss_df[clean_inlier_idx, 1] = 0
                #     loss_df[mislabel_outlier_idx, 1] = -1
                #     loss_df[clean_outlier_idx, 1] = 2
                #     loss_df = pd.DataFrame(loss_df, columns = ['loss', 'class'])
                #     print('saving early loss class')
                #     loss_df.to_csv('early_loss_class_satellite.csv')

                    # clean_inlier_loss = loss_std_clean_values[clean_inlier_idx]
                    # mislabel_inlier_loss = loss_std_clean_values[mislabel_inlier_idx]
                    # clean_outlier_loss = loss_std_clean_values[clean_outlier_idx]
                    # mislabel_outlier_loss = loss_std_clean_values[mislabel_outlier_idx]
                ########################################################################################

                # outlier_loss_clean_list = sorted(list(zip(outlier_loss_idx, outlier_loss_values)), key=lambda x: x[1])
                # outlier_loss_bad_list = sorted(list(zip(outlier_loss_idx, outlier_loss_values)), key=lambda x: -x[1])
                # outlier_loss_clean_idx = [x[0] for x in outlier_loss_clean_list]
                # outlier_loss_bad_idx = [x[0] for x in outlier_loss_bad_list]

                # std_list = sorted(enumerate(early_loss_std),key=lambda x: x[1])
                # inlier_clean_data_idx = inlier_loss_clean_idx[:int(clean_data_len/2)]
                tmp_inlier_bad_data_idx = tmp_inlier_loss_bad_idx[:int(bad_data_len / 2)]
                # outlier_clean_data_idx = outlier_loss_clean_idx[:int(clean_data_len / 4)]
                print(f'---In the selected clean data, outlier/inlier %: {(num_add_outlier/num_add_inlier) * 100}---')
                tmp_outlier_bad_data_idx = tmp_outlier_loss_bad_idx[:int(bad_data_len / 2)]

                print(f'---There are in total {len(clean_idx)} selected clean data---')
                clean_data = full_data[clean_idx, :]

                # for i in clean_data:
                #     retrieval_pool_clean.append(i)
                #
                # clean_pool = np.array(retrieval_pool_clean)

                bad_data_idx = np.union1d(tmp_inlier_bad_data_idx, tmp_outlier_bad_data_idx)
                print(len(bad_data_idx))
                bad_data = new_train_clean_bad_set[bad_data_idx, :]
                bad_test_loader = DataLoader(dataset=MyDataSet(bad_data), batch_size=1, shuffle=False)

            ##################################################################################
            else:
                loss_clean_list = sorted(enumerate(early_loss_avg_norm), key=lambda x: x[1])
                loss_clean_idx = [x[0] for x in loss_clean_list]
                tmp_clean_data = new_train_clean_bad_set[loss_clean_idx, :]
                loss_clean_true_idx = list(set([instance2idx[tuple(i)] for i in tmp_clean_data[:,:-1]]))
                # tmp_clean_data_idx = loss_clean_idx[:clean_data_len]
                # # print(len(clean_data_idx))
                # tmp_clean_data = new_train_clean_bad_set[tmp_clean_data_idx, :]
                # tmp_clean_data_true_idx = list(set([instance2idx[tuple(i)] for i in tmp_clean_data[:,:-1]])) # corresponding clean data idx in index in full data

                num_add = 0
                idx = 0
                print(f'---retriving top {clean_data_len} clean data---')
                while num_add < clean_data_len:
                    if loss_clean_true_idx[idx] not in clean_idx:
                        clean_idx.append(loss_clean_true_idx[idx])
                        num_add += 1
                        tmp = full_data[loss_clean_true_idx[idx], :]
                        tmp_clean = np.append(tmp, 0)
                        retrieval_pool.append(tmp_clean)
                    idx += 1

                # union clean data in each iteration
                clean_data = full_data[clean_idx, :]
                print(f"---There are in total {len(clean_data)} selected clean data---")

                # extract the top n early loss sample as potential mislabel data
                loss_std_bad_list = sorted(enumerate(early_loss_avg_norm), key=lambda x: -x[1])
                loss_std_bad_idx = [x[0] for x in loss_std_bad_list]
                bad_data_idx = loss_std_bad_idx[:bad_data_len]
                bad_data = new_train_clean_bad_set[bad_data_idx, :]

                # print(f'---retriving top {bad_data_len} bad data---')
                # for i in bad_data:
                #     tmp = i
                #     tmp_bad = np.append(tmp, 2)
                #     retrieval_pool.append(tmp_bad)

        ####################################################################################################################################################################
        print('#####running on clean data#####')
        print(f'---In the selected clean data, outlier/inlier %: {(len(np.where(clean_data[:,-1] == 1)[0]) / len(np.where(clean_data[:,-1] == 0)[0])) * 100}---')
        model_clean = run_NN(clean_data, data_length=len(clean_data), epoch=args.train_epoch, return_loss=False,
                                   converge=True, entropyStop=args.earlyStop, weight_sample=False, correct_label=False, run_twin_net=False, noisy_rate=noisy_rate)
        # further filter bad data in potential clean pool
        # predict_logits = inference_NN(model_clean, new_train_clean_bad_set, device=device, return_label=False)
        # predict_proba = np.argmax(predict_logits, axis=1)
        # conf_sample_idx = np.where(predict_proba > 0.999)[0]
        # clean_idx
        ####################################################################################################################################################################
        loss_function_bad = nn.CrossEntropyLoss()
        influence_bad = []
        bad_test_loader = DataLoader(dataset=MyDataSet(bad_data), batch_size=1, shuffle=False)
        model_clean.eval()
        print('#####fit on dirty data#####')
        num = 0
        for data in tqdm(bad_test_loader, leave=False):
            _, train_feature_bad, train_label_bad = data
            train_feature_bad = train_feature_bad.to(device)
            train_label_bad = train_label_bad.to(device)
            train_label_predict_bad = model_clean(train_feature_bad)

            # GPU加速
            train_label_predict_bad = train_label_predict_bad.to(device)
            train_loss_bad = loss_function_bad(train_label_predict_bad, train_label_bad)
            # train_loss_bad = LabelSmoothingCrossEntropy(train_label_predict_bad, train_label_bad)

            # influence function
            train_loss_bad_gradient = torch.autograd.grad(train_loss_bad, model_clean.parameters(), allow_unused=True)

            grad = 0.0
            idx = bad_data_idx[num]
            for item in train_loss_bad_gradient:
                if item == None:
                    continue
                item = torch.norm(item)
                grad += item
            num+=1

            tmp = [idx, grad]
            influence_bad.append(tmp)

        influence_bad_sorted = sorted(influence_bad, key=lambda x: -x[1])
        influence_bad_idx = [x[0] for x in influence_bad_sorted]
        print(f"length of bad data pool {len(influence_bad_idx)}")
        print(influence_bad_idx)

        annotated_bad_data = new_train_clean_bad_set[influence_bad_idx[:detect_num],:]
        print(f'---retriving top {detect_num} bad data---')
        for i in annotated_bad_data:
            tmp = i
            tmp_bad = np.append(tmp, 1) # 1 indicates bad data
            retrieval_pool.append(tmp_bad)

        ##################################################################################
        correct_num = 0
        detect_idx_50 = []
        # 每轮detect500个
        print(f'---Retriving the top {detect_num} bad samples---')
        for i in range(min(detect_num, len(influence_bad_idx))):
            total_detect_num += 1
            detect_idx_50.append(influence_bad_idx[i])
            if (total_len - 1) >= influence_bad_idx[i] >= clean_len:
                correct_num += 1
                detected_bad_labels.append(new_train_clean_bad_set[influence_bad_idx[i], -1])
                detected_instance = new_train_clean_bad_set[influence_bad_idx[i], :-1]
                detected_idx = instance2idx[tuple(detected_instance)]
                true_bad_detected_idx.append(detected_idx)

        print("loss最高的脏数据占比为:{}".format(correct_num / detect_num))
        total_correct_num += correct_num

        # 计算总的精度
        print("第{}轮的precision score {}, recall score {}".format(times + 1, total_correct_num / (detect_num * (times + 1)),
                                                    total_correct_num / len(bad_data_full)))

        ground_truth_tmp = []
        new_train_clean_bad_set = []

        ######################################################################################################################
        # 重新构建训练集
        for i in range(total_len):
            if i not in detect_idx_50:
                new_train_clean_bad_set.append(train_clean_bad_set_copy[i])

        train_clean_bad_set_copy = copy.deepcopy(new_train_clean_bad_set)

        total_len = len(new_train_clean_bad_set)
        bad_len = bad_len - correct_num
        clean_len = total_len - bad_len
        print(f"clean data length: {clean_len}, bad data length: {bad_len}, total length: {total_len}")

        new_train_clean_bad_set = np.array(new_train_clean_bad_set)
        noisy_label = np.squeeze(new_train_clean_bad_set[:,-1]).astype(np.int64)
        # model_final = run_NN(new_train_clean_bad_set, len(new_train_clean_bad_set), epoch = 40)
        train_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=128, shuffle=True)
        test_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=1, shuffle=False)
        ######################################################################################################################

    print('detected mislabel outliers / detected total mislabel percentage %:',
          (np.sum(mislabel_labels) / total_correct_num) * 100)

    precision = total_correct_num / total_detect_num
    print(total_correct_num, len(train_bad_set))
    recall = total_correct_num / len(train_bad_set)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    print('f1 score', f1, 'precision', precision, 'recall', recall)
    # print('---------------------------')
    outlier_mislabel_num = np.sum(detected_bad_labels)
    detected_mislabel_outlier_percentage = (outlier_mislabel_num/len(detected_bad_labels)) * 100
    record_df.loc[name, :] = [precision, recall, f1, detected_mislabel_outlier_percentage]
    print(record_df)
    print('outlier mislabel num', np.sum(detected_bad_labels), '%', np.sum(detected_bad_labels)/len(detected_bad_labels))

    return true_bad_detected_idx

    #############################################################################################################
    # retrieval_pool_test = []
    # remain_data_true_idx = list(set([instance2idx[tuple(i)] for i in new_train_clean_bad_set[:, :-1]]))
    # repeat_clean_idx = np.intersect1d(clean_idx, remain_data_true_idx)
    # mask = np.ones(len(new_train_clean_bad_set), dtype=bool)
    # for idx in repeat_clean_idx:
    #     # 找到 new_train_clean_bad_set 中对应 idx 的行
    #     row_idx = np.where([instance2idx[tuple(i)] == idx for i in new_train_clean_bad_set[:, :-1]])[0]
    #     mask[row_idx] = False  # 标记为 False，表示需要删除
    #
    # # 使用布尔掩码筛选未标注的数据
    # unannoted_data = new_train_clean_bad_set[mask,:]
    #
    # print('重复数据长度:', len(repeat_clean_idx))
    # print('未标注数据长度:', len(unannoted_data))
    #
    # correct_num = 0
    # for i in unannoted_data:
    #     tmp = i
    #     tmp = np.append(tmp, 999)
    #     retrieval_pool_test.append(tmp)
    # print(f'retrieved annotated data {len(retrieval_pool)}, retrieved unannotated data {len(retrieval_pool_test)}')
    #
    # train_pool = copy.deepcopy(retrieval_pool + retrieval_pool_test)
    # features = []
    # labels = []
    # for item in train_pool:
    #     features.append(item[:-1])
    #     labels.append(item[-1])
    # features = pd.DataFrame(features)
    # from sklearn.neighbors import NearestNeighbors
    # nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree').fit(features)
    #
    # train_retrieval_pool = []
    # for i in range(len(retrieval_pool)): #对每一个被抽取的data
    #     distance, index = nbrs.kneighbors(retrieval_pool[i][:-1].reshape(1, -1), 10) #抽取10个距离最近的instances
    #     neighbors = [] # 10个knn点
    #     neighbors_labels = []
    #     for x in np.array(index.tolist()[0]):
    #         neighbors.append(torch.from_numpy(train_pool[x][:-1]))
    #         neighbors_labels.append(torch.tensor(train_pool[x][-1]))
    #     # neighbors_labels = torch.tensor(neighbors_labels)
    #
    #     # 第一个knn neighbor的feature
    #     input_features = neighbors[0].reshape([1, len(neighbors[0])])
    #     id = 1
    #     while id < len(neighbors): #拼接全部10个knn的feature
    #         input_features = torch.cat([input_features, neighbors[id].reshape([1, len(neighbors[id])])], 0)
    #         id += 1
    #     # 利用attention将10个knn点的特征进行聚合
    #     weighted_sum = aggregation_feature(input_features)
    #
    #     # 利用interaction layer将10个knn点的特征和标签进行聚合
    #     aggregated = interactions_feature_label(neighbors, neighbors_labels)
    #
    #     # 特征聚合：knn特征和原始数据特征，标签为原始数据的标签
    #     train_feature_retrieval = torch.cat((torch.from_numpy(retrieval_pool[i][:-1]), weighted_sum, aggregated), 0)
    #     train_label_retrieval = torch.tensor(retrieval_pool[i][-1])
    #     tmp_retrieval = [train_feature_retrieval.detach(), train_label_retrieval]
    #
    #     train_retrieval_pool.append(tmp_retrieval)
    # # ***************************************
    #
    # # 收集KNN测试数据集***************************************
    # test_retrieval_pool = []
    # retrieval_pool_test_true_idx = [instance2idx[tuple(i[:-2])] for i in retrieval_pool_test]
    # for i in range(len(retrieval_pool_test)): #对每一个retrieval_test_pool里的data
    #     # 对于retrieval pool test里的每一个instance找出它的10个knn instance并聚合他们的features
    #     distance, index = nbrs.kneighbors(retrieval_pool_test[i][:-1].reshape(1, -1), 10)
    #     neighbors = []
    #     neighbors_labels = []
    #     for x in np.array(index.tolist()[0]):
    #         neighbors.append(torch.from_numpy(train_pool[x][:-1]))
    #         neighbors_labels.append(torch.tensor(train_pool[x][-1]))
    #     # neighbors_labels = torch.tensor(neighbors_labels)
    #     input_features = neighbors[0].reshape([1, len(neighbors[0])])
    #     id = 1
    #     while id < len(neighbors): #将10个knn点特征和label进行拼接
    #         input_features = torch.cat([input_features, neighbors[id].reshape([1, len(neighbors[id])])], 0)
    #         id += 1
    #     # 利用attention将10个knn点样本的特征进行聚合
    #     weighted_sum = aggregation_feature(input_features)
    #     # 利用interaction layer将10个knn点样本的特征和标签进行聚合
    #     aggregated = interactions_feature_label(neighbors, neighbors_labels)
    #
    #     #将数据点和其10个knn点的聚合特征进行拼接---特征增强
    #     test_feature_retrieval = torch.cat((torch.from_numpy(retrieval_pool_test[i][:-1]), weighted_sum, aggregated), 0)
    #     test_label_retrieval = torch.tensor(retrieval_pool_test[i][-1])
    #     tmp_retrieval = [test_feature_retrieval.detach(), test_label_retrieval] #标签为假999
    #     test_retrieval_pool.append(tmp_retrieval)
    # # ***************************************
    #
    # # 训练retrieval_pool---------------------------------
    # train_loader_retrieval = DataLoader(dataset=MyDataSet_retrieval(train_retrieval_pool), batch_size=128, shuffle=True)
    # # test_loader_retrieval = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=1, shuffle=False)
    # model_retrieval = Model_retrieval(3 * full_data.shape[1])
    # model_retrieval = model_retrieval.double().to(device)
    # optimizer_retrieval = torch.optim.Adam(model_retrieval.parameters(), lr=0.001)
    # loss_function_retrieval = nn.MSELoss()
    # loss_function_retrieval = loss_function_retrieval.to(device)
    # model_retrieval.train()
    # for i in range(20):
    #     for data in tqdm(train_loader_retrieval, leave=True):
    #         # GPU加速
    #         train_feature, train_label = data
    #         train_feature = train_feature.double().to(device)
    #         train_label = train_label.double().to(device)
    #         optimizer_retrieval.zero_grad()
    #         train_label_predict = model_retrieval(train_feature)
    #
    #         # GPU加速
    #         train_label_predict = train_label_predict.to(device)
    #         train_loss = loss_function_retrieval(train_label_predict.squeeze().double(), train_label.double())
    #         train_loss.backward()
    #         optimizer_retrieval.step()
    # # 训练retrieval_pool结束，测试retrieval_pool开始---------------------------------
    # test_loader_retrieval = DataLoader(dataset=MyDataSet_retrieval(test_retrieval_pool), batch_size=1, shuffle=False)
    # model_retrieval.eval()
    # with torch.no_grad():
    #     num = 0
    #     hh = 0
    #     for data in test_loader_retrieval:
    #         test_feature, _ = data
    #         test_feature = test_feature.double().to(device)
    #         test_label_predict = model_retrieval(test_feature)
    #         if retrieval_pool_test_true_idx[num] not in clean_idx:
    #             hh += 1
    #         if test_label_predict > 0.9:
    #             if retrieval_pool_test_true_idx[num] not in clean_idx:
    #                 # print(num, ":", test_label_predict, len(clean_idx))
    #                 total_detect_num += 1
    #                 if (len(full_data) - 1) >= retrieval_pool_test_true_idx[num] >= len(train_clean_set):
    #                     print(len(full_data) - 1, retrieval_pool_test_true_idx[num], len(train_clean_set))
    #                     correct_num += 1
    #         num += 1
    # # 测试retrieval_pool结束***************************************
    # print(len(clean_idx), num, hh)
    # total_correct_num += correct_num
    #
    # print('total correct number', total_correct_num, 'total detect number', total_detect_num)
    # # 计算总的精度
    # print("最终轮的precision{},recall{}".format(total_correct_num / total_detect_num,
    #                                             total_correct_num / len(bad_data_full)))
    #
    # precision = total_correct_num / total_detect_num
    # recall = total_correct_num / len(bad_data_full)
    # f1 = 2 * (precision * recall) / (precision + recall)
    # # print("detect_iterate * detect_num:{}".format(detect_iterate * detect_num))
    # print("detect precision：{}".format(precision))
    # print("detect recall：{}".format(recall))
    # print("detect f1 score：{}".format(f1))


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # if the loss entropy decreases for 3 successive epochs, break
    parser.add_argument('--early_detect_k', type=int, default=50, help='patience parameter for early loss detect')
    parser.add_argument('--converge_k', type=int, default=100, help='patience parameter for converge training')
    parser.add_argument('--train_epoch', type=int, default=40, help='training epoch on clean data') # 40 epochs
    parser.add_argument('--detect_epoch', type=int, default=5, help='epochs in each early loss detect iteration') # 3,5,10
    parser.add_argument('--twin_net_epoch', type=int, default=10, help='epochs in twin net loss detection') # 10 epochs is not good
    # parser.add_argument('--warm_up_epoch', type=int, default=15, help='warm up epoch on noisy data')
    parser.add_argument('--detect_iterate', type=int, default=5, help='total detect iterations')
    parser.add_argument('--Rdown', type=float, default=0.01, help='loss entropy decreasing percentage')
    parser.add_argument('--n_eval', type=int, default=512, help='number of evaluation data on early stopping') #512
    parser.add_argument('--converge_batch_size', type=int, default=64, help='train converge batch size') # should set to large batch size
    parser.add_argument('--detect_batch_size', type=int, default=128, help='detect batch size')
    parser.add_argument('--lr_detect', type=float, default=0.01, help='learning rate for early loss detection')
    parser.add_argument('--lr_converge', type=float, default=0.001, help='learning rate for converge training')
    parser.add_argument('--noisy_rate_inlier', type=float, default=0.017, help='noisy rate for inlier')
    parser.add_argument('--noisy_rate_outlier', type=float, default=0.75, help='noisy rate for outlier')
    parser.add_argument('--earlyStop', action='store_true') # 'EntropyAE' or 'VanillaAE'
    parser.add_argument('--method', default="misdetect", type=str, help='method to use')
    args = parser.parse_args()

    detect_iterate = args.detect_iterate

    data_dir = '../../lof_predictions/predict'
    data_list = os.listdir(data_dir)
    print(data_list)
    record_df = pd.DataFrame(columns=['precision', 'recall', 'f1', 'detected_mislabel_outlier_percentage'])
    feature_extract_dict = {}
    # 读取 CSV 文件
    for name in data_list:
        if name == 'cover.csv':
            continue
        # for iter_num in detect_iterate:
        # else:
        if name == ('Pendigits.csv'):
            print(name)
            df_real = pd.read_csv(f'../../lof_predictions/real/{name}')
            df_lof = pd.read_csv(f'../../lof_predictions/predict/{name}')
            # print(df_lof.shape, df_real.shape)
            # df_lof = df_lof.drop_duplicates().reset_index(drop=True)
            # df_real = df_real.drop_duplicates().reset_index(drop=True)
            print(df_lof.shape, df_real.shape)
            good_sample_ratio = np.sum(df_real['label'] == df_lof['label']) / len(df_real)
            noisy_rate = 1 - good_sample_ratio
            mislabel_num = np.sum(df_real['label'] != df_lof['label'])
            mislabel_labels = df_lof.iloc[np.where(df_real['label'] != df_lof['label'])[0],-1]
            print('mislabel outliers / total mislabel instances have percentage:', (np.sum(mislabel_labels)/mislabel_num) * 100)

            # estimate mislabel ratio
            mislabel_range = [(0,0.1), (0.1, 0.3), (0.3, 0.4)]
            for low_bound, up_bound in mislabel_range:
                if low_bound <= 1 - good_sample_ratio <= up_bound:
                    detect_num = int(((low_bound + up_bound) / 2)/5 * len(df_lof))
                else:
                    detect_num = int(mislabel_num / detect_iterate)

            print(f"######Detect {detect_num} each iteration#####")
            columns = df_lof.columns.tolist()
            classes = len(df_lof.iloc[:, -1].unique())
            label_index = len(columns) - 1

            train_clean_bad_set, train_clean_set, train_bad_set, ground_truth, noisy_label = train_clean_dataset_builder(df_real, df_lof)

            if args.method == 'misdetect':
                detect_result = misdetect(full_data=train_clean_bad_set, clean_data=train_clean_set, bad_data=train_bad_set,
                                       ground_truth=ground_truth, detect_num=int(mislabel_num / detect_iterate), detect_iterate=args.detect_iterate,
                                       clean_data_len=int(len(train_clean_set) / 4) ,
                                       bad_data_len=int(len(train_bad_set)/ 4), early_stop=args.earlyStop,
                                       separate_inlier_outlier=False, weight_sampler=False, run_twin_net=True, noisy_rate=noisy_rate)

            if args.method == 'coteaching':
                # run coteaching
                model_1, model_2 = Model_loss(), Model_loss()
                coteaching(model_1, model_2, device, train_clean_bad_set, train_clean_set, train_bad_set)


            if args.method == 'cleanlab':
                # run cleanlab
                cleanLab(len(train_clean_bad_set), len(train_clean_set), train_clean_bad_set)

            if args.method == 'coteaching_plus':
                model_1, model_2 = Model_loss(), Model_loss()
                coteaching_plus(model_1, model_2, device, train_clean_bad_set, train_clean_set, train_bad_set)




