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


def train_clean_dataset_builder(df_real, df_lof):
    """
    df_real: dataframe for real outlier dataset with al true labels
    df_lof: dataframe for lof predicted outlier dataset with some mislabels
    """
    # df_lof = df_lof.drop_duplicates()
    # df_real = df_real.drop_duplicates()
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
    train_set_raw = np.concatenate([X, predicted_labels], axis = 1)
    print(train_set.shape)

    ground_truth = np.array(df_real.iloc[:,-1]).reshape(-1,1)
    good_idx_array = np.where(predicted_labels == ground_truth)[0]
    all_idx_array = np.array(range(len(df_lof)))
    bad_idx_array = np.setdiff1d(all_idx_array, good_idx_array)

    train_clean_set = train_set[good_idx_array, :]
    train_clean_set_raw = train_set_raw[good_idx_array, :]
    train_bad_set = train_set[bad_idx_array, :]
    train_bad_set_raw = train_set_raw[bad_idx_array, :]
    train_clean_bad_set = np.concatenate((train_clean_set, train_bad_set), axis = 0)  # we put mislabeled set under clean set to later verify if detected instance is truly mislabeled
    train_clean_bad_set_raw = np.concatenate((train_clean_set_raw, train_bad_set_raw), axis=0)  # we put mislabeled set under clean set to later verify if detected instance is truly mislabeled
    print(train_clean_set.shape, train_bad_set.shape)


    ground_truth = df_real.iloc[:,-1].values
    ground_truth = np.concatenate((ground_truth[good_idx_array], ground_truth[bad_idx_array]), axis = 0)
    print(ground_truth.shape)
    # print(train_clean_bad_set)
    return train_clean_bad_set, train_clean_set, train_bad_set, ground_truth, train_clean_bad_set_raw


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
torch.manual_seed(0) # cpu seed
torch.cuda.manual_seed(0) # gpu seed
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

# kk = 768
# class Model_retrieval(nn.Module):
#     def __init__(self, input_len):
#         super(Model_loss, self).__init__()
#         self.net1 = Sequential(OrderedDict([
#             ('batch norm1', nn.BatchNorm1d(input_len, momentum=0.99)),
#             ('linear1', nn.Linear(input_len, kk)),
#             ('relu1', nn.ReLU())
#         ]))
#         self.net2 = Sequential(OrderedDict([
#             ('batch norm2', nn.BatchNorm1d(kk, momentum=0.99)),
#             ('linear2', nn.Linear(kk, kk)),
#             ('relu2', nn.ReLU())
#         ]))
#         self.net3 = Sequential(OrderedDict([
#             ('batch norm3', nn.BatchNorm1d(kk, momentum=0.99)),
#             ('linear3', nn.Linear(kk, classes)),
#             ('sigmoid', nn.Sigmoid())
#         ]))
#
#     def forward(self, x):
#         a = self.net1(x)
#         b = self.net2(a) + a
#         c = self.net3(b)
#         return c


# class Model_loss(nn.Module):
#     def __init__(self):
#         super(Model_loss, self).__init__()
#         self.fc1 = nn.Linear(label_index, 50)
#         self.bn1 = nn.BatchNorm1d(50)
#         self.fc2 = nn.Linear(50, 100)
#         self.bn1 = nn.BatchNorm1d(100)
#         self.fc3 = nn.Linear(100,50)
#         self.bn1 = nn.BatchNorm1d(50)
#         self.fc4 = nn.Linear(50,classes)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         out = self.fc4(x)
#         return out


k = 320
k2 = 384

class Model_loss(nn.Module):
    def __init__(self):
        super(Model_loss, self).__init__()
        self.net1 = Sequential(OrderedDict([
            ('batch norm1', nn.BatchNorm1d(label_index, momentum=0.99)),
            ('linear1', nn.Linear(label_index, k)),
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
            ('linear3', nn.Linear(k, classes)),
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



class NegEntropy(object):
    """
    negative entropy penalty for over-confident prediction
    """
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


def inference_NN(model, test_data, device = device, return_label = False):
    test_dataloader = DataLoader(MyDataSet(test_data), batch_size=128, shuffle=False)
    model.eval()
    predict_logits = []
    for data in test_dataloader:
        _, test_feature, test_label = data
        test_feature = test_feature.to(device)
        test_label = test_label.to(device)
        net_out = model(test_feature)
        predict_logits.append(net_out.data.cpu().numpy())

    if return_label:
        predict_proba = [F.softmax(torch.tensor(logits), dim=1).numpy() for logits in predict_logits]
        predicted_labels = np.argmax(np.concatenate(predict_proba, axis=0), axis=1)
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



def run_NN(data, data_length, epoch = 3, return_loss=False, converge = False, entropyStop = False,
           n_eval=100, k =100, R_down=0.1, weight_sample = False):

    if converge: # train model until converge use larger batch
        # train epoch on clean data
        epoch = args.train_epoch
        entropyStop = args.earlyStop # whether to use entropy early stop

        learning_rate, converge_batch_size =args.lr_converge, args.converge_batch_size

        # parameters for entropy early stop
        k, R_down, n_eval = args.converge_k, args.Rdown, args.n_eval

        # whether do entropy based early stop
        if entropyStop:
            ES = ModelEntropyEarlyStop(k=k, R_down=R_down)
            N_eval = min(n_eval, data.shape[0])
            eval_index = np.random.choice(data.shape[0], N_eval, replace=False)  # random sample evaluation set
            data_eval = data[eval_index]
            isStop = False
            train_loader = DataLoader(dataset=MyDataSet(data), batch_size=converge_batch_size, shuffle=True)
            eval_loader = DataLoader(dataset=MyDataSet(data_eval), batch_size=128, shuffle=False)

        else:
            train_loader = DataLoader(dataset=MyDataSet(data), batch_size=converge_batch_size, shuffle=True)
            test_loader = DataLoader(dataset=MyDataSet(data), batch_size=1, shuffle=False)


    else: # run early loss detection
        epoch = args.detect_epoch
        learning_rate, detect_batch_size = args.lr_detect, args.detect_batch_size
        train_loader = DataLoader(dataset=MyDataSet(data), batch_size=detect_batch_size, shuffle=True)
        test_loader = DataLoader(dataset=MyDataSet(data), batch_size=128, shuffle=False)

    model = Model_loss() # use MLP model
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss() # use to collect batch average loss
    loss_function_raw = nn.CrossEntropyLoss(reduction='none') # collect per sample loss

    model.train()
    early_loss = np.zeros((data_length, epoch), dtype=np.float64)
    predict_proba = np.zeros((data_length, epoch), dtype=np.float64)

    for epo in range(epoch):
        for data in tqdm(train_loader, leave=True, dynamic_ncols=True):
            # GPU加速
            _, train_feature, train_label = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer.zero_grad()
            train_label_predict = model(train_feature)
            # GPU加速
            train_label_predict = train_label_predict.to(device)
            train_loss = loss_function(train_label_predict, train_label)

            train_loss.backward()
            optimizer.step()

            if entropyStop:
                isStop = ES.step(eval_loader, model, n_eval) # do entropy early stop iteratively
                if isStop:
                    break
            if entropyStop and isStop: # stop condition
                break

        if return_loss: # collect early loss from each epoch

            model.eval()
            with torch.no_grad():
                num = 0
                correct_num = 0
                for data in test_loader:
                    idx, test_feature, test_label = data
                    test_feature = test_feature.to(device)
                    test_label = test_label.to(device)
                    test_label_predict = model(test_feature)
                    loss = loss_function_raw(test_label_predict, test_label).detach().cpu().numpy()
                    early_loss[idx, epo] = loss
                    # predict_proba[num, epo] = test_label_predict[0][test_label[0]]
                    # if torch.argmax(test_label_predict[0]) == torch.argmax(test_label):
                    #     correct_num += 1
                    num += 1

    if return_loss: # do early loss detection (only specify to true if using single net early loss detection)
        if entropyStop: # if use entropy stop in early loss detetction
            return isStop, early_loss, predict_proba
        else:
            return None, early_loss, predict_proba

    else: # train on clean data phase

        if entropyStop: # return the best model from enrtopstopping
            model_best = ES.getBestModel()  # load the best model
            return model_best

        else: # return model after finishing all epochs
            return model



# 测试influence detect的效果
def misdetect(full_data, clean_data, bad_data, ground_truth, detect_num, detect_iterate,
              clean_data_len, bad_data_len, early_stop = True, separate_inlier_outlier = False, weight_sampler = False):
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

        isStop, early_loss, predict_proba = run_NN(new_train_clean_bad_set, data_length=len(new_train_clean_bad_set), epoch=args.detect_epoch, return_loss=True,
                                                  converge=False, entropyStop=False, weight_sample=False)

        ##################################################################################
        # use avergae early loss
        scaler = MinMaxScaler(feature_range=(0, 1))
        early_loss_avg = np.squeeze(np.mean(early_loss, axis = 1)).reshape(-1,1)
        early_loss_std = np.squeeze(np.std(early_loss, axis = 1)).reshape(-1,1)
        predict_proba_avg = np.squeeze(np.mean(predict_proba, axis = 1)).reshape(-1,1)

        # use EMA early loss
        ema_early_loss = calculate_ema_loss(early_loss, alpha=0.9)[:,-1]
        ema_early_loss_norm = (ema_early_loss - np.min(ema_early_loss)) / (np.max(ema_early_loss) - np.min(ema_early_loss))  # scale the early loss and std
        ema_early_loss_std = np.squeeze(np.std(calculate_ema_loss(early_loss), axis=1))
        ema_early_loss_std_norm = (ema_early_loss_std - np.min(ema_early_loss_std)) / (np.max(ema_early_loss_std) - np.min(ema_early_loss_std))

        early_loss_avg_norm = scaler.fit_transform(early_loss_avg)
        early_loss_std_norm = scaler.fit_transform(early_loss_std)

        # loss_list = list(enumerate(early_loss_avg_norm))
        loss_list = list(enumerate(ema_early_loss_norm ))
        loss_values = np.squeeze(np.array([x[1] for x in loss_list]))
        loss_idx = np.array([x[0] for x in loss_list])
        # loss_list = list(enumerate(early_loss_avg_norm))
        # loss_values = np.squeeze(np.array([x[1] for x in loss_list]))
        # loss_idx = np.array([x[0] for x in loss_list])
        ##################################################################################
        if separate_inlier_outlier:
            print('#####running separate inlier outlier loss detection#####')
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
            print(f'---retriving top {int(9*clean_data_len/10)} inlier clean data---')
            while num_add_inlier < int(9*clean_data_len / 10):
                if idx >= len(outlier_clean_true_idx):
                    break
                if inlier_clean_true_idx[idx] not in clean_idx:
                    clean_idx.append(inlier_clean_true_idx[idx])
                    num_add_inlier += 1
                    tmp = full_data[inlier_clean_true_idx[idx], :]
                    tmp_clean = np.append(tmp, 0) # 0 indicates clean data
                    retrieval_pool.append(tmp_clean)
                idx += 1

            num_add_outlier = 0
            idx = 0
            print(f'---retriving top {int(clean_data_len/10)} outlier clean data---')
            while num_add_outlier < int(clean_data_len / 10):
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
            if times == 0:
                loss_df = np.zeros((total_len, 2))
                loss_df[:, 0] = loss_values
                mislabel_inlier_idx = tmp_inlier_loss_idx[np.where(tmp_inlier_loss_idx >= clean_len)[0]]
                clean_inlier_idx = tmp_inlier_loss_idx[np.where(tmp_inlier_loss_idx < clean_len)[0]]

                mislabel_outlier_idx = tmp_outlier_loss_idx[np.where(tmp_outlier_loss_idx >= clean_len)[0]]
                clean_outlier_idx = tmp_outlier_loss_idx[np.where(tmp_outlier_loss_idx < clean_len)[0]]
                print('length of mislabel inlier', len(mislabel_inlier_idx))

                print('length of mislabel outlier', len(mislabel_outlier_idx))
                assert len(mislabel_inlier_idx) + len(mislabel_outlier_idx) == mislabel_num

                loss_df[mislabel_inlier_idx, 1] = 1
                loss_df[clean_inlier_idx, 1] = 0
                loss_df[mislabel_outlier_idx, 1] = -1
                loss_df[clean_outlier_idx, 1] = 2
                loss_df = pd.DataFrame(loss_df, columns = ['loss', 'class'])
                print('saving early loss class')
                loss_df.to_csv(f'loss_dir/early_loss_class_{name}', index = False)
            ########################################################################################

            # outlier_loss_clean_list = sorted(list(zip(outlier_loss_idx, outlier_loss_values)), key=lambda x: x[1])
            # outlier_loss_bad_list = sorted(list(zip(outlier_loss_idx, outlier_loss_values)), key=lambda x: -x[1])
            # outlier_loss_clean_idx = [x[0] for x in outlier_loss_clean_list]
            # outlier_loss_bad_idx = [x[0] for x in outlier_loss_bad_list]

            # std_list = sorted(enumerate(early_loss_std),key=lambda x: x[1])
            # inlier_clean_data_idx = inlier_loss_clean_idx[:int(clean_data_len/2)]
            tmp_inlier_bad_data_idx = tmp_inlier_loss_bad_idx[:int(bad_data_len / 2)]
            # outlier_clean_data_idx = outlier_loss_clean_idx[:int(clean_data_len / 4)]
            # print(f'---In the selected clean data, outlier/inlier %: {(num_add_outlier/num_add_inlier) * 100}---')
            tmp_outlier_bad_data_idx = tmp_outlier_loss_bad_idx[:int(bad_data_len / 2)]

            print(f'---There are in total {len(clean_idx)} selected clean data---')
            clean_data = full_data[clean_idx, :]

            # for i in clean_data:
            #     retrieval_pool_clean.append(i)
            #
            # clean_pool = np.array(retrieval_pool_clean)

            bad_data_idx = np.union1d(tmp_inlier_bad_data_idx, tmp_outlier_bad_data_idx).astype(int)
            print(bad_data_idx)
            bad_data = new_train_clean_bad_set[bad_data_idx, :]
            bad_test_loader = DataLoader(dataset=MyDataSet(bad_data), batch_size=1, shuffle=False)

        ##################################################################################
        else:
            loss_clean_list = sorted(enumerate(ema_early_loss_norm), key=lambda x: x[1])
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
                if idx >= len(loss_clean_true_idx):
                    break
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
            loss_std_bad_list = sorted(enumerate(ema_early_loss_norm), key=lambda x: -x[1])
            loss_std_bad_idx = [x[0] for x in loss_std_bad_list]
            bad_data_idx = loss_std_bad_idx[:bad_data_len]

            # print(f'---retriving top {bad_data_len} bad data---')
            # for i in bad_data:
            #     tmp = i
            #     tmp_bad = np.append(tmp, 2)
            #     retrieval_pool.append(tmp_bad)

        ####################################################################################################################################################################
        print('#####running on clean data#####')
        print(f'---In the selected clean data, outlier/inlier %: {(len(np.where(clean_data[:,-1] == 1)[0]) / len(np.where(clean_data[:,-1] == 0)[0])) * 100}---')
        model_clean = run_NN(clean_data, data_length=len(clean_data), epoch=args.train_epoch, return_loss=False,
                                   converge=True, entropyStop=args.earlyStop, weight_sample=False)


        bad_data = new_train_clean_bad_set[bad_data_idx, :]

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
        # print(influence_bad_idx)

        annotated_bad_data = new_train_clean_bad_set[influence_bad_idx[:detect_num],:]
        print(f'---retriving top {detect_num} bad data---')
        for i in annotated_bad_data:
            tmp = i
            tmp_bad = np.append(tmp, 1) # 1 indicates bad data
            retrieval_pool.append(tmp_bad)

        influence_bad_label = [np.squeeze(new_train_clean_bad_set[idx, -1]) for idx in influence_bad_idx]
        print('number of outliers in retrieved bad pool', np.sum(influence_bad_label))
        print('number of inliers in retrieced bad pool', len(influence_bad_idx) - np.sum(influence_bad_label))

        ##################################################################################
        correct_num = 0
        detect_idx_50 = []
        # 每轮detect detect_num 个错误标签样本
        print(f'---Retriving the top {detect_num} bad samples---')
        for i in range(min(detect_num, len(influence_bad_idx))):
            detect_idx_50.append(influence_bad_idx[i])
            total_detect_num +=1
            detected_instance = new_train_clean_bad_set[influence_bad_idx[i], :-1]
            detected_idx = instance2idx[tuple(detected_instance)]
            true_bad_detected_idx.append(detected_idx)

            if (total_len - 1) >= influence_bad_idx[i] >= clean_len:
                correct_num += 1
                detected_bad_labels.append(new_train_clean_bad_set[influence_bad_idx[i], -1])


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


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--early_detect_k', type=int, default=50, help='patience parameter for early loss detect')
    parser.add_argument('--converge_k', type=int, default=100, help='patience parameter for converge training')
    parser.add_argument('--train_epoch', type=int, default=40, help='training epoch on clean data') # 40 epochs
    parser.add_argument('--detect_epoch', type=int, default=5, help='epochs in each early loss detect iteration') # 3,5,10
    parser.add_argument('--detect_iterate', type=int, default=5, help='total detect iterations')
    parser.add_argument('--Rdown', type=float, default=0.01, help='loss entropy decreasing percentage')
    parser.add_argument('--n_eval', type=int, default=512, help='number of evaluation data on early stopping') #512
    parser.add_argument('--converge_batch_size', type=int, default=128, help='train converge batch size') # should set to large batch size
    parser.add_argument('--detect_batch_size', type=int, default=128, help='detect batch size')
    parser.add_argument('--lr_detect', type=float, default=0.01, help='learning rate for early loss detection') # 0.01
    parser.add_argument('--lr_converge', type=float, default=0.001, help='learning rate for converge training') # 0.001
    parser.add_argument('--earlyStop', action='store_true') # 'EntropyAE' or 'VanillaAE'
    parser.add_argument('--method', default="misdetect", type=str, help='method to use')
    args = parser.parse_args()

    detect_iterate = args.detect_iterate

    data_dir = '../../lof_predictions/predict'
    data_list = os.listdir(data_dir)
    print(data_list)
    record_df = pd.DataFrame(columns=['precision', 'recall', 'f1', 'detected_mislabel_outlier_percentage'])

    # 读取 CSV 文件
    for name in data_list:

        if (name == 'cover.csv') or (name == 'shuttle.csv') or  (name == 'SpamBase.csv'):
            continue

        if name == ('Thursday.csv'):
            print(name)
            df_real = pd.read_csv(f'../../lof_predictions/real/{name}')
            df_lof = pd.read_csv(f'../../lof_predictions/predict/{name}')
            # print(df_lof.shape, df_real.shape)
            # df_lof = df_lof.drop_duplicates().reset_index(drop=True)
            # df_real = df_real.drop_duplicates().reset_index(drop=True)
            print(df_real.head())
            print(df_lof.shape, df_real.shape)
            good_sample_ratio = np.sum(df_real['label'] == df_lof['label']) / len(df_real)
            noisy_rate = 1 - good_sample_ratio
            mislabel_num = np.sum(df_real['label'] != df_lof['label'])
            mislabel_labels = df_lof.iloc[np.where(df_real['label'] != df_lof['label'])[0],-1]
            print('mislabel outliers / total mislabel instances have percentage:', (np.sum(mislabel_labels)/mislabel_num) * 100)

            outlier_num = np.sum(df_lof['label'] == 1)
            inlier_num = np.sum(df_lof['label'] == 0)
            outlier_noisy_rate = (mislabel_num/2) / outlier_num
            inlier_noisy_rate = (mislabel_num/2) / inlier_num
            print('outlier noisy rate:', outlier_noisy_rate, 'inlier noisy rate:', inlier_noisy_rate)

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

            train_clean_bad_set, train_clean_set, train_bad_set, ground_truth, train_clean_bad_set_raw = train_clean_dataset_builder(df_real, df_lof)

            if args.method == 'misdetect':
                mislabel_idx = misdetect(full_data=train_clean_bad_set, clean_data=train_clean_set, bad_data=train_bad_set,
                                          ground_truth=ground_truth, detect_num=int(mislabel_num / detect_iterate), detect_iterate=args.detect_iterate,
                                          clean_data_len=int(len(train_bad_set) / 4) + 10,
                                          bad_data_len=int(len(train_bad_set)/ 4), early_stop=args.earlyStop,
                                          separate_inlier_outlier=0, weight_sampler=False)

                mislabel_idx = np.array(mislabel_idx)
                train_clean_bad_set_raw[mislabel_idx, -1] = 1 - train_clean_bad_set_raw[mislabel_idx, -1]
                corrected_data = pd.DataFrame(train_clean_bad_set_raw, columns=columns)
                print('saving data...')
                corrected_data.to_csv(f'../../annotated_data_{name}', index=False)

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