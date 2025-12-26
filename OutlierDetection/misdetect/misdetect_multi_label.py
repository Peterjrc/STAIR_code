from email.policy import default

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import Normalizer, RobustScaler, MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
import torch.nn.init as init
from EntropyEarlyStop import ModelEntropyEarlyStop, cal_entropy
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

columns = {"eeg": ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'Eye'],
           "uscensus": ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-num', 'Marital-status', 'Occupation',
                        'Relationship', 'Race', 'Sex', 'Capital-gain', ' Capital-loss', 'Hours-per-week',
                        'Native-country', 'Income'],
           "credit": ['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'age',
                      'NumberOfTime30-59DaysPastDueNotWorse',
                      'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                      'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'],
           "hotel": ['Booking_ID', 'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
                     'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 'lead_time',
                     'arrival_year',
                     'arrival_month', 'arrival_date', 'market_segment_type', 'repeated_guest',
                     'no_of_previous_cancellations',
                     'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests',
                     'booking_status'],
           "heart": ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR',
                     'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'HeartDisease'],
           "wine": ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
                    'free_sulfur_dioxide',
                    'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality', 'color'],
           "airline": ['id', 'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
                       'Inflight wifi service',
                       'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink',
                       'Online boarding',
                       'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
                       'Baggage handling', 'Checkin service',
                       'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
                       'satisfaction'],
           "mobile": ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
                      'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width',
                      'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi', 'price_range'],
           "covertype": ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                         'Vertical_Distance_To_Hydrology',
                         'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                         'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2',
                         'Wilderness_Area3', 'Wilderness_Area4',
                         'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
                         'Soil_Type7', 'Soil_Type8', 'Soil_Type9',
                         'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
                         'Soil_Type16',
                         'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
                         'Soil_Type23',
                         'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29',
                         'Soil_Type30',
                         'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
                         'Soil_Type37',
                         'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type'],
           }

classes = {"eeg": ['0', '1'],
           "uscensus": ['0', '1'],
           "credit": ['0', '1'],
           "hotel": ['0', '1'],
           "heart": ['0', '1'],
           "wine": ['0', '1', '2', '3', '4', '5', '6'],
           "airline": ['0', '1'],
           "mobile": ['0', '1', '2', '3'],
           "covertype": ['0', '1', '2', '3', '4', '5', '6'],
           }

labels = {"eeg": "Eye",
          "uscensus": "Income",
          "credit": "SeriousDlqin2yrs",
          "hotel": "booking_status",
          "heart": "HeartDisease",
          "wine": "quality",
          "airline": "satisfaction",
          "mobile": "price_range",
          "covertype": "Cover_Type",
          }


def mis_injection(dataset, mis_rate, mis_distribution):
    label_index = columns[dataset].index(labels[dataset])

    # clean数据占比
    good_sample_ratio = 1 - float(mis_rate)

    train_set_before = pd.read_csv("/home/dengyuhao/misdetect/dataset/" + dataset + "/" + dataset + "_normalize.csv")
    if dataset in ["credit", "airline"]:
        train_set_before = train_set_before.fillna(axis=1, method='ffill')
        train_set_before = train_set_before.dropna()
        # 首先将pandas读取的数据转化为array
        train_set_before = np.array(train_set_before)
        train_set_before = np.delete(train_set_before, 0, axis=1)

    train_set_before = np.array(train_set_before)
    label = np.array(train_set_before).T[label_index]
    train_set_before = np.delete(train_set_before, label_index, axis=1)

    # 将train_set转化为特定list形式
    train_set = []
    for i in range(len(train_set_before)):
        tmp = [torch.tensor(train_set_before[i].tolist()), int(label[i])]
        train_set.append(tmp)

    train_set_tmp = copy.deepcopy(train_set)

    if mis_distribution == "random":
        cnt_label = {}
        for idx, tensor in enumerate(train_set):
            cnt_label[tensor[1]] = cnt_label.get(tensor[1], 0) + 1
        print(len(cnt_label))

        cnt_good_label_tgt = {}
        for k, v in cnt_label.items():
            cnt_good_label_tgt[k] = int(v * good_sample_ratio)

        manipulate_label = {}
        good_idx_set = []
        for idx, tensor in enumerate(train_set):
            manipulate_label[tensor[1]] = manipulate_label.get(tensor[1], 0) + 1
            if manipulate_label[tensor[1]] > cnt_good_label_tgt[tensor[1]]:
                p = np.random.randint(0, len(cnt_label))
                while True:
                    if p != tensor[1]:
                        train_set[idx][1] = p
                        break
                    p = np.random.randint(0, len(cnt_label))
            else:
                good_idx_set.append(idx)

        good_idx_array = np.array(good_idx_set)
        all_idx_array = np.arange(len(train_set))
        bad_idx_array = np.setdiff1d(all_idx_array, good_idx_array)
        train_clean_dataset = []
        for i in good_idx_array:
            train_clean_dataset.append(train_set[i])
            if train_set[i][1] != train_set_tmp[i][1]:
                print("--------------------------------")
        train_bad_dataset = []
        for i in bad_idx_array:
            train_bad_dataset.append(train_set[i])
            if train_set[i][1] == train_set_tmp[i][1]:
                print("--------------------------------")

        train_bad_dataset = []
        for i in bad_idx_array:
            train_bad_dataset.append(train_set_tmp[i])

        train_clean_bad_set_ground_truth = train_clean_dataset + train_bad_dataset

        train_clean_bad_set = train_clean_dataset + train_bad_dataset
        print(len(train_clean_dataset), len(train_bad_dataset), len(train_clean_bad_set))
        return train_clean_dataset, train_bad_dataset, train_clean_bad_set, train_clean_bad_set_ground_truth

    else:
        # ---------------------------------------------------------------
        # 随机制造脏数据，而不是每个类取固定比例制造脏数据
        train_clean_size = int(good_sample_ratio * len(train_set_tmp))
        train_bad_size = len(train_set_tmp) - train_clean_size
        train_clean_set, train_bad_set = torch.utils.data.random_split(train_set_tmp,
                                                                       [train_clean_size, train_bad_size])

        train_set = []
        for i in train_set_tmp:
            train_set.append(list(i))

        train_clean_dataset = []
        train_bad_dataset = []
        for i in train_bad_set:
            train_bad_dataset.append(list(i))

        for i in train_clean_set:
            train_clean_dataset.append(list(i))

        train_clean_bad_set_ground_truth = train_clean_dataset + train_bad_dataset

        for i in train_bad_dataset:
            p = np.random.randint(0, len(classes))
            while True:
                if p != i[1]:
                    i[1] = p
                    break
                p = np.random.randint(0, len(classes))

        train_clean_bad_set_ground_truth = train_clean_dataset + train_bad_dataset
        train_clean_bad_set = train_clean_dataset + train_bad_dataset
        print(len(train_clean_dataset), len(train_bad_dataset), len(train_clean_bad_set))
        return train_clean_dataset, train_bad_dataset, train_clean_bad_set, train_clean_bad_set_ground_truth


def multi_class_dataset_builder(data_dir, dataset, noisy_rate, mis_distribution):
    train_set_before = pd.read_csv(os.path.join(data_dir, dataset), sep=',')

    if dataset == 'wine_normalize.csv':
        col1_data = train_set_before.pop('quality')  # 移除并获取 col1 数据
        col1_index = train_set_before.columns.get_loc('color')  # 获取 col2 的索引
        train_set_before.insert(col1_index + 1, 'quality', col1_data)  # 在 col2 后插入 col1

    # print(train_set_before)
    columns = train_set_before.columns.tolist()
    label_index = len(columns) - 1
    good_sample_ratio = 1 - noisy_rate
    classes = len(np.unique(train_set_before.iloc[:, label_index]))
    print(classes)

    train_set_before = np.array(train_set_before)
    ground_truth = train_set_before[:, label_index].astype(int)
    label_min, label_max = ground_truth.min(), ground_truth.max()
    if label_min > 0:
        label_adjust = label_min
    else:
        label_adjust = 0

    ground_truth = ground_truth - label_adjust

    # 处理 train_set
    train_set = []
    for i in range(len(train_set_before)):
        tmp = np.hstack((train_set_before[i, :-1], int(ground_truth[i])))  # 拼接 features 和 label
        train_set.append(tmp)
    # print(train_set)
    train_set = np.array(train_set)  # 转换为 NumPy 数组
    train_set_tmp = copy.deepcopy(train_set)

    if mis_distribution == "random":
        train_clean_size = int(good_sample_ratio * len(train_set_before))
        train_bad_size = len(train_set_before) - train_clean_size
        train_clean_dataset, train_bad_dataset = train_test_split(train_set, train_size=train_clean_size,
                                                                random_state=42,  # 保证可复现
                                                                shuffle=True)

        train_clean_set = train_clean_dataset
        train_bad_set = train_bad_dataset
        train_clean_bad_set_ground_truth = np.concatenate((train_clean_set, train_bad_set), axis=0)

        for i in train_bad_set:
            p = np.random.randint(0, classes)
            while True:
                if p != i[-1]:
                    i[-1] = p
                    break
                p = np.random.randint(0, classes)
        print(train_clean_set.shape, train_bad_set.shape)

        train_clean_bad_set = np.concatenate((train_clean_set, train_bad_set), axis=0)

    else: # equal injection
    # 计算每个类别的样本数
        cnt_label = {label: (ground_truth == label).sum() for label in np.unique(ground_truth)}
        cnt_good_label_tgt = {k: int(v * good_sample_ratio) for k, v in cnt_label.items()}
        print(cnt_label)
        manipulate_label = {}
        good_idx_set = []

        for idx, data in enumerate(train_set):
            label = int(data[-1])  # 获取当前样本的标签
            manipulate_label[label] = manipulate_label.get(label, 0) + 1

            if manipulate_label[label] > cnt_good_label_tgt[label]:
                # 随机更改标签
                p = np.random.randint(0, len(cnt_label))
                while p == label:
                    p = np.random.randint(0, len(cnt_label))
                train_set[idx, -1] = p  # 修改 NumPy 数组中的标签
            else:
                good_idx_set.append(idx)

        good_idx_array = np.array(good_idx_set)
        all_idx_array = np.arange(len(train_set))
        bad_idx_array = np.setdiff1d(all_idx_array, good_idx_array)
        # 构造干净和污染的数据集
        # scale the raw data
        # X = train_set[:, :-1].values
        # X_transformed = scaler.fit_transform(X)
        # train_set = np.concatenate([X_transformed, noisy_label], axis = 1)

        train_clean_set = train_set[good_idx_array]
        train_bad_set = train_set[bad_idx_array]
        print(train_clean_set.shape, train_bad_set.shape)

        train_clean_bad_set = np.concatenate((train_clean_set, train_bad_set), axis=0)  # 拼接 NumPy 数组

    return train_clean_bad_set, train_clean_set, train_bad_set, ground_truth, classes, columns


def convert_to_soft_labels(hard_labels, num_classes, smoothing=0.1):
    confidence = 1.0 - 2 * smoothing
    smooth_prob = smoothing / (num_classes - 1)  # Distribute the remaining probability mass

    # Create one-hot encoding of the hard labels
    one_hot_labels = OneHotEncoder(categories='auto').fit_transform(hard_labels.reshape(-1, 1)).toarray()
    # Apply label smoothing
    # print(one_hot_labels)
    soft_labels = one_hot_labels * confidence + smooth_prob
    # print(soft_labels)
    return soft_labels


def min_max_normalize(data, feature_range=(0, 1)):
    """
    Min-Max归一化，将数据缩放到指定范围。

    Args:
        data (array-like): 输入数据，NumPy数组或列表。
        feature_range (tuple): 目标范围 (min, max)，默认是 (0, 1)。

    Returns:
        np.ndarray: 归一化后的数据。
    """
    data = np.array(data)  # 转换为NumPy数组
    min_val, max_val = feature_range
    data_min = data.min()
    data_max = data.max()
    normalized = (data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
    return normalized


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
        return feature, label


k = 320
torch.manual_seed(0)
torch.cuda.manual_seed(0)


# questions to answer
# 1. why use batchnorm  facilitate training
# 2. why design simple MLP
# 3. why apply softmax
# 4. why skip connection
# 5. need to use SGD optimizer
# 6. large small batch comparison

def he_init_weights(m):
    if isinstance(m, nn.Linear):  # 检查是否是线性层
        init.kaiming_normal_(m.weight, nonlinearity='relu')  # He 正态初始化
        if m.bias is not None:
            init.zeros_(m.bias)  # 偏置初始化为 0


# from torch_geometric.nn import MLP
#
# class Autoencoder(nn.Module):
#     def __init__(self,in_dim):
#         super(Autoencoder, self).__init__()
#         h_dim = 64
#         self.lins = MLP([in_dim,h_dim,in_dim],dropout=0.2)
#
#     def forward(self,x):
#         x_ = self.lins(x)
#         return torch.sum(torch.square(x_ - x),dim=-1)


class Model_shallow(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net1 = Sequential(OrderedDict([
            # ('batch norm1', nn.BatchNorm1d(label_index, momentum=0.99)),
            ('linear1', nn.Linear(label_index, k)),
            ('relu1', nn.ReLU())
        ]))
        self.net2 = Sequential(OrderedDict([
            # ('batch norm2', nn.BatchNorm1d(k, momentum=0.99)),
            ('linear2', nn.Linear(k, k)),
            ('relu2', nn.ReLU())
        ]))
        self.net3 = Sequential(OrderedDict([
            # ('batch norm3', nn.BatchNorm1d(k, momentum=0.99)),
            ('linear3', nn.Linear(k, classes)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        a = self.net1(x)
        b = self.net2(a) + a
        c = self.net3(b)
        return c


k = 320
k2 = 384


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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


def inference_NN(model, test_data, device=device):
    test_dataloader = DataLoader(MyDataSet(test_data), batch_size=100, shuffle=False)
    model.eval()
    predict_proba = []
    for data in test_dataloader:
        test_feature, test_label = data
        test_feature = test_feature.to(device)
        test_label = test_label.to(device)
        net_out = model(test_feature)
        predict_proba.append(F.softmax(net_out, dim=1).data.cpu().numpy())
    return np.concatenate(predict_proba)


def run_NN(data, data_length, epoch=3, return_loss=False, converge=False, entropyStop=False,
           n_eval=100, k=100, R_down=0.1, weight_sample=False):
    if converge:  # train model until converge use larger batch
        epoch = args.train_epoch
        entropyStop = args.earlyStop  # whether to use entropy early stop
        learning_rate, converge_batch_size = args.lr, args.converge_batch_size
        k, R_down, n_eval = args.converge_k, args.Rdown, args.n_eval

        # whether do entropy based early stop
        if entropyStop:
            ES = ModelEntropyEarlyStop(k=k, R_down=R_down)
            N_eval = min(n_eval, data.shape[0])
            eval_index = np.random.choice(data.shape[0], N_eval, replace=False)  # random sample evaluation set
            data_eval = data[eval_index]
            isStop = False
            train_loader = DataLoader(dataset=MyDataSet(data), batch_size=converge_batch_size, shuffle=True)
            eval_loader = DataLoader(dataset=MyDataSet(data_eval), batch_size=1, shuffle=False)

        else:
            train_loader = DataLoader(dataset=MyDataSet(data), batch_size=converge_batch_size, shuffle=True)

        # whether do class weight sampling
        if weight_sample:
            sampler = weight_sampler(data=data, noisy_label=data[:, -1])
            train_loader = DataLoader(dataset=MyDataSet(data), batch_size=converge_batch_size, shuffle=False,
                                      sampler=sampler)

        else:
            train_loader = DataLoader(dataset=MyDataSet(data), batch_size=converge_batch_size, shuffle=True)
            # eval_loader = DataLoader(dataset=MyDataSet(data_eval), batch_size=1, shuffle=False)


    else:  # run early loss detection
        epoch = args.detect_epoch
        learning_rate, detect_batch_size = args.lr, args.detect_batch_size
        # k, R_down, n_eval = args.early_detect_k, args.Rdown, args.n_eval
        # if entropyStop:
        #     ES = ModelEntropyEarlyStop(k=k, R_down=R_down)
        #     N_eval = min(n_eval, data.shape[0])
        #     eval_index = np.random.choice(data.shape[0], N_eval, replace=False)  # random sample evaluation set
        #     data_eval = data[eval_index]
        #     isStop = False
        #
        #     train_loader = DataLoader(dataset=MyDataSet(data), batch_size=detect_batch_size, shuffle=True)
        #     eval_loader = DataLoader(dataset=MyDataSet(data_eval), batch_size=1, shuffle=False)
        #     test_loader = DataLoader(dataset=MyDataSet(data), batch_size=1, shuffle=False)
        # else:
        if weight_sample:
            sampler = weight_sampler(data=data, noisy_label=data[:, -1])
            train_loader = DataLoader(dataset=MyDataSet(data), batch_size=detect_batch_size, shuffle=False,
                                      sampler=sampler)
            test_loader = DataLoader(dataset=MyDataSet(data), batch_size=1, shuffle=False)

        else:
            train_loader = DataLoader(dataset=MyDataSet(data), batch_size=detect_batch_size, shuffle=True)
            test_loader = DataLoader(dataset=MyDataSet(data), batch_size=1, shuffle=False)

    model = Model()  # use MLP model
    # model = Autoencoder(latent_dim=latent_dim).to(device) # use Autoencoder model
    model = model.to(device)
    # model.apply(he_init_weights)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()
    # loss_function = loss_function.to(device)

    model.train()
    early_loss = np.zeros((data_length, epoch), dtype=np.float64)
    predict_proba = np.zeros((data_length, epoch), dtype=np.float64)
    # early_loss_per_label = [[[], [], [], [], []] for _ in range(total_len)]  # store recent few epochs loss

    for epo in range(epoch):  # train for few epochs
        for data in tqdm(train_loader, leave=True, dynamic_ncols=True):
            # GPU加速
            train_feature, train_label = data
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
                isStop = ES.step(eval_loader, model, n_eval)  # iteratively
                if isStop:
                    break
            if entropyStop and isStop:
                break

        if return_loss:  # collect early loss from each epoch
            model.eval()
            with torch.no_grad():
                num = 0
                correct_num = 0
                for data in test_loader:
                    test_feature, test_label = data
                    test_feature = test_feature.to(device)
                    test_label = test_label.to(device)
                    test_label_predict = model(test_feature)
                    # recon_score = model(test_feature)
                    # test_label_predict = test_label_predict.to(device)
                    # loss = LabelSmoothingCrossEntropy(test_label_predict, test_label)
                    loss = loss_function(test_label_predict, test_label)
                    # loss = recon_score.cpu().detach().numpy()
                    # loss_per_label = loss_function.get_per_class_loss(test_label_predict, test_label).cpu().numpy()
                    # print(loss_per_label)
                    early_loss[num, epo] = loss
                    # print(test_label_predict[0][test_label[0]])
                    predict_proba[num, epo] = test_label_predict[0][test_label[0]]
                    if torch.argmax(test_label_predict[0]) == torch.argmax(test_label):
                        correct_num += 1
                    # early_loss_per_label[num][epo].append(loss_per_label) # store recent few epochs loss
                    num += 1

    if return_loss:
        if entropyStop:  # if use entropy stop in early loss detetction
            return isStop, early_loss, predict_proba
        else:
            return None, early_loss, predict_proba
    else:
        if entropyStop:  # return the best model from enrtopstopping
            model_best = ES.getBestModel()  # load the best model
            return model_best
        else:  # return model after finishing all epochs
            return model


# 测试influence detect的效果
def misdetect(full_data, clean_data, bad_data, ground_truth, detect_num, detect_iterate,
              clean_data_len, bad_data_len, early_stop=True, separate_inlier_outlier=False, weight_sampler=False):
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
    new_train_clean_bad_set = copy.deepcopy(full_data)  # used for the first iteration
    X_full = full_data[:, :-1]
    y_full = full_data[:, -1]
    noisy_label = np.squeeze(new_train_clean_bad_set[:, -1]).astype(np.int64)  # used to correct noisy label
    train_clean_bad_set_copy = copy.deepcopy(full_data)  # used to construct new_train_clean_bad_set for next iteration
    instance2idx = {tuple(j): i for i, j in enumerate(full_data[:, :-1])}  # used to match instance to idx
    bad_data_full = bad_data
    # 总共的correct个数
    count = 0
    total_correct_num = 0
    threshold_good = 0.99
    threshold_bad = 0.01

    for times in range(detect_iterate):

        count += 1
        # total_len = len(new_train_clean_bad_set)
        class_idx = {}
        for c in range(classes):
            class_idx[c] = np.where(noisy_label == c)[0]

        print('---running early loss detection---')
        isStop, early_loss, predict_proba = run_NN(new_train_clean_bad_set, data_length=len(new_train_clean_bad_set),
                                                   epoch=args.detect_epoch, return_loss=True,
                                                   converge=False, entropyStop=False, weight_sample=False)

        ##################################################################################
        # average loss based on window size of 3
        scaler = MinMaxScaler(feature_range=(0, 1))
        early_loss_sum = sorted(
            enumerate(np.squeeze(scaler.fit_transform(np.mean(early_loss, axis=1).reshape(-1, 1) * 10))),
            key=lambda x: -x[1])
        early_loss_avg = np.squeeze(np.mean(early_loss, axis=1)).reshape(-1, 1)
        early_loss_std = np.squeeze(np.std(early_loss, axis=1)).reshape(-1, 1)
        predict_proba_avg = np.squeeze(np.mean(predict_proba, axis=1)).reshape(-1, 1)

        # scale the early loss and std
        early_loss_avg_norm = scaler.fit_transform(early_loss_avg)
        early_loss_std_norm = scaler.fit_transform(early_loss_std)

        loss_std_clean_list = sorted(
            enumerate(early_loss_avg_norm / ((predict_proba_avg+1e-12) * (early_loss_std_norm + 1e-12))), key=lambda x: x[1])
        # loss_std_clean_list = sorted(
        #     enumerate(early_loss_avg_norm), key=lambda x: x[1])
        loss_list = list(enumerate(early_loss_avg_norm / (early_loss_std_norm + 1e-12)))
        loss_idx = np.array([x[0] for x in loss_list])
        loss_values = np.array([x[1] for x in loss_list])

        loss_std_clean_values = np.array([x[1] for x in loss_std_clean_list])
        loss_std_clean_idx = np.array([x[0] for x in loss_std_clean_list])

        ##################################################################################
        if separate_inlier_outlier:
            class_loss = {}
            bad_data_idx = set()
            clean_data_idx = set()
            for c in range(classes):
                class_loss_idx = loss_idx[class_idx[c]]
                class_loss_values = loss_values[class_loss_idx]
                class_loss_clean_list = sorted(list(zip(class_loss_idx, class_loss_values)), key=lambda x: x[1])
                class_loss_bad_list = sorted(list(zip(class_loss_idx, class_loss_values)), key=lambda x: -x[1])
                class_loss_clean_idx = [x[0] for x in class_loss_clean_list]
                class_loss_bad_idx = [x[0] for x in class_loss_bad_list]


            # std_list = sorted(enumerate(early_loss_std),key=lambda x: x[1])
                class_clean_data_idx = class_loss_clean_idx[:int(clean_data_len / classes)]
                class_bad_data_idx = class_loss_bad_idx[:int(bad_data_len / classes)]

                clean_data_idx.update(class_clean_data_idx)
                bad_data_idx.update(class_bad_data_idx)

            clean_data_idx = np.array(list(clean_data_idx))
            bad_data_idx = np.array(list(bad_data_idx))
            print(bad_data_idx)
            clean_data = new_train_clean_bad_set[clean_data_idx, :]
            bad_data = new_train_clean_bad_set[bad_data_idx, :]
            bad_test_loader = DataLoader(dataset=MyDataSet(bad_data), batch_size=1, shuffle=False)

        ##################################################################################
        else:
            clean_data_idx = loss_std_clean_idx[:clean_data_len]
            print(len(clean_data_idx))
            # clean_data_idx = loss_std_clean_idx[:clean_data_len] # select top n clean data
            clean_data = new_train_clean_bad_set[clean_data_idx, :]
            # clean_train_loader = DataLoader(dataset=MyDataSet(clean_data), batch_size=64, shuffle=True)
            # clean_test_loader = DataLoader(dataset=MyDataSet(clean_data), batch_size=64, shuffle=False)

            # extract the top n early loss sample as potential mislabel data
            loss_std_bad_list = sorted(enumerate(early_loss_avg_norm), key=lambda x: -x[1])
            loss_std_bad_idx = [x[0] for x in loss_std_bad_list]
            loss_std_bad_values = [x[1] for x in loss_std_bad_list]
            bad_data_idx = loss_std_bad_idx[:bad_data_len]
            bad_data = new_train_clean_bad_set[bad_data_idx, :]
            bad_test_loader = DataLoader(dataset=MyDataSet(bad_data), batch_size=1, shuffle=False)

        ##################################################################################
        print('#####running on clean data#####')
        model_clean = run_NN(clean_data, data_length=len(clean_data), epoch=args.train_epoch, return_loss=False,
                             converge=True, entropyStop=args.earlyStop, weight_sample=False)

        ##################################################################################
        # build a classifier
        # clf_X = SVC(gamma='auto', probability=True, random_state=0)
        # X_clf = new_train_clean_bad_set[:, :-1]
        # y_clf = new_train_clean_bad_set[:, -1]
        # X_full = full_data[:, :-1]
        # clf_X.fit(X_clf, y_clf)
        # clf_predict_proba_X = clf_X.predict_proba(X_full)[:, 1]
        #
        # SVM_threshold = 0.5
        # print("F-1 score from SVM:",
        #       metrics.f1_score(ground_truth, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
        # print("precision from SVM:",
        #       metrics.precision_score(ground_truth, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
        # print("recall from SVM:",
        #       metrics.recall_score(ground_truth, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
        #
        # SVM_threshold = np.sort(clf_predict_proba_X)[::-1][int(np.sum(ground_truth))]
        # print("F-1 score from SVM:",
        #       metrics.f1_score(ground_truth, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
        # print("precision from SVM:",
        #       metrics.precision_score(ground_truth, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
        # print("recall from SVM:",
        #       metrics.recall_score(ground_truth, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))

        ##################################################################################
        # print('#####adding confident samples#####')
        # predict_proba = inference_NN(model_clean, new_train_clean_bad_set)
        # noisy_label = np.squeeze(new_train_clean_bad_set[:,-1]).astype(np.int64)
        # class_proba = predict_proba[np.arange(len(noisy_label)), noisy_label]
        # predict_bad_idx = np.where(class_proba < threshold_bad)[0]
        # predict_good_idx = np.where(class_proba > threshold_good)[0]
        # print(predict_bad_idx, predict_good_idx)
        # new_bad_idx = np.setdiff1d(predict_bad_idx, bad_data_idx)
        # new_good_idx = np.setdiff1d(predict_good_idx, clean_data_idx)
        # print(f'new dirty data: {len(new_bad_idx)}, new clean data: {len(new_good_idx)}')
        #
        # # add new bad data
        # if len(new_bad_idx) > 0:
        #     bad_data_idx.append(new_bad_idx)
        # # remove new clean data
        # if len(new_good_idx) > 0:
        #     for idx in new_good_idx:
        #         if idx in bad_data_idx:
        #             bad_data_idx.remove(idx)

        #################################################################################
        loss_function_bad = nn.CrossEntropyLoss()
        influence_bad = []
        model_clean.eval()
        print('#####fit on dirty data#####')
        num = 0
        for data in tqdm(bad_test_loader, leave=False):
            train_feature_bad, train_label_bad = data
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
            num += 1

            tmp = [idx, grad]
            influence_bad.append(tmp)

        influence_bad_sorted = sorted(influence_bad, key=lambda x: -x[1])
        influence_bad_idx = [x[0] for x in influence_bad_sorted]
        print(f"length of bad data pool {len(influence_bad_idx)}")
        print(influence_bad_idx)

        ##################################################################################
        correct_num = 0
        true_bad_detected_idx = []

        detect_idx_50 = []
        early_loss_predicted = np.zeros((total_len, 1), dtype=np.int32)
        early_loss_actual = np.ones((total_len, 1), dtype=np.int32)
        for i in range(clean_len):
            early_loss_actual[i][0] = 0

        # 每轮detect500个
        for i in range(min(detect_num, len(influence_bad_idx))):
            detect_idx_50.append(influence_bad_idx[i])
            early_loss_predicted[influence_bad_idx[i]][0] = 1

            if (total_len - 1) >= influence_bad_idx[i] >= clean_len:
                correct_num += 1
                detected_bad_labels.append(new_train_clean_bad_set[influence_bad_idx[i], -1])
                true_bad_detected_idx.append(influence_bad_idx[i])

        print("loss最高的脏数据占比为:{}".format(correct_num / detect_num))
        total_correct_num += correct_num

        # 计算总的精度
        print("第{}轮的precision score {}, recall score {}".format(times + 1,
                                                                   total_correct_num / (detect_num * (times + 1)),
                                                                   total_correct_num / len(bad_data_full)))

        ground_truth_tmp = []
        new_train_clean_bad_set = []

        ###########################################################
        # predict_proba = inference_NN(model_clean, full_data)[:, 1]
        # print("F-1 score from NN on full data:",
        #       metrics.f1_score(ground_truth, np.array([int(i) for i in predict_proba > 0.5]), average='macro'))
        # print("Precision score from NN on full data:",
        #       metrics.precision_score(ground_truth, np.array([int(i) for i in predict_proba > 0.5])))
        # print("Recall score from NN on full data:",
        #       metrics.recall_score(ground_truth, np.array([int(i) for i in predict_proba > 0.5])))
        ###########################################################

        for i in range(total_len):
            if i not in detect_idx_50:
                new_train_clean_bad_set.append(train_clean_bad_set_copy[i])

        train_clean_bad_set_copy = copy.deepcopy(new_train_clean_bad_set)

        total_len = len(new_train_clean_bad_set)
        bad_len = bad_len - correct_num
        clean_len = total_len - bad_len
        print(f"clean data length: {clean_len}, bad data length: {bad_len}, total length: {total_len}")

        new_train_clean_bad_set = np.array(new_train_clean_bad_set)
        noisy_label = np.squeeze(new_train_clean_bad_set[:, -1]).astype(np.int64)
        train_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=1, shuffle=False)
        ###########################################################

        # loss_bad = enumerate(list(early_loss))
        # bad_pool_idx = [x[0] for x in loss_std_list]  # 获取排序好后b坐标,下标在第0位
        # std_idx = np.array([x[0] for x in std_list])
        # std_value = scaler.fit_transform(np.array([x[1] for x in std_list]).reshape(-1,1))
        # potential_bad_idxes_std = std_value[potential_bad_idxes]
        # potential_bad_idxes_loss = loss_values[potential_bad_idxes]
        # loss_std_bad_list = sorted(enumerate(potential_bad_idxes_loss / ((potential_bad_idxes_std + 1e-8))), key=lambda x: -x[1])
        # loss_std_bad_idxes = [x[0] for x in loss_std_bad_list]
        # print(loss_std_bad_idxes)
        #
        # potential_good_idxes_std = std_value[potential_good_idxes]
        # potential_good_idxes_loss = loss_values[potential_good_idxes]
        # loss_std_good_idxes = sorted(enumerate(potential_good_idxes_loss / (potential_good_idxes_std+1e-8)), key=lambda x: -x[1])
        # loss_std_good_idxes = [x[0] for x in loss_std_good_idxes]
        # print(loss_std_good_idxes)
        #
        # bad_inliers_set = np.concatenate((new_train_clean_bad_set[bad_inliers_idx, :-1], np.full((len(new_train_clean_bad_set[bad_inliers_idx, :-1],)), 1)), axis = 1)
        # good_inliers_set = np.concatenate((new_train_clean_bad_set[good_inliers_idx, :-1], np.full((len(new_train_clean_bad_set[good_inliers_idx, :-1],)), 0)), axis = 1)
        # temp_train_set = np.concatenate((good_inliers_set, bad_inliers_set), axis = 0)

        # bad_pool_idx = [x for x in early_loss_per_label.keys()]  # 获取排序好后b坐标,下标在第0位
        # print(len(bad_pool_idx))
    precision = total_correct_num / (detect_num * detect_iterate)
    print(total_correct_num, len(train_bad_set))
    recall = total_correct_num / len(train_bad_set)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    print('f1 score', f1, 'precision', precision, 'recall', recall)
    # print('---------------------------')
    record_df.loc[name, :] = [precision, recall, f1]

    # 将二维列表转换为 DataFrame
    # features = [item[0].numpy() for item in train_clean_bad_set_modify]
    # labels = np.array([item[1] for item in train_clean_bad_set_modify])
    # X_df = pd.DataFrame(features, columns=[f'attr{i}' for i in range(1, len(features[0]) + 1)])
    # y_df = pd.DataFrame(labels, columns=['label'])
    # df_new = pd.concat((X_df, y_df), axis=1)
    # cleaned_df= pd.DataFrame(df_new)

    # features = [item[0].numpy() for item in train_clean_bad_tmp]
    # original_labels = np.array([item[1] for item in train_clean_bad_tmp])
    # loss = [item[2] for item in train_clean_bad_tmp]
    # # print(loss)
    # annotation = np.array([item[3] for item in train_clean_bad_tmp])
    #
    # X_df = pd.DataFrame(features, columns=[f'attr{i}' for i in range(1, len(features[0]) + 1)])
    # original_labels_df = pd.DataFrame(original_labels, columns=['original_label'])
    # loss_df = pd.DataFrame(loss, columns=[f'loss{epo}' for epo in range(1, epoch+1)])
    # annotation_df = pd.DataFrame(annotation, columns=['annotation'])
    #
    # df_all_feature = pd.concat((X_df, original_labels_df, loss_df, annotation_df), axis=1)
    # 指定要保存的 CSV 文件路径
    # csv_file_path1 = f'../../lof_predictions/modify/{name}'
    # # csv_file_path1 = 'auto_table.csv'

    # # 将 DataFrame 写入 CSV 文件
    # cleaned_df.to_csv(csv_file_path1, index=False)
    # final_dirty_set_csv.to_csv(csv_file_path1, index=False, header=repair_column)


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # if the loss entropy decreases for 3 successive epochs, break
    parser.add_argument('--early_detect_k', type=int, default=50, help='patience parameter for early loss detect')
    parser.add_argument('--converge_k', type=int, default=100, help='patience parameter for converge training')
    parser.add_argument('--train_epoch', type=int, default=10, help='training epoch on clean data')
    parser.add_argument('--detect_epoch', type=int, default=3, help='epochs in each detect iteration')
    parser.add_argument('--detect_iterate', type=int, default=5, help='total detect iteration')
    parser.add_argument('--Rdown', type=float, default=0.01, help='loss entropy decreasing percentage')
    parser.add_argument('--n_eval', type=int, default=512, help='number of evaluation data')  # 512
    parser.add_argument('--converge_batch_size', type=int, default=64,
                        help='train converge batch size')  # should set to large batch size
    parser.add_argument('--detect_batch_size', type=int, default=256, help='detect batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--earlyStop', action='store_true')  # 'EntropyAE' or 'VanillaAE'
    parser.add_argument('--noisy_rate', type=float, default=0.2, help='set the noisy rate')
    args = parser.parse_args()

    detect_iterate = args.detect_iterate
    multi_label_detect = 1

    if multi_label_detect:
        data_dir = '../../../FRL/falling_rule_list/data'
        data_list = ['wine_normalize.csv', 'hotel_normalize.csv', 'uscensus_normalize.csv']
        train_clean_bad_set, train_clean_set, train_bad_set, ground_truth, classes, columns = multi_class_dataset_builder(
            data_dir=data_dir, dataset='wine_normalize.csv', noisy_rate=args.noisy_rate, mis_distribution='equal')
        good_sample_ratio = len(train_clean_set) / len(train_clean_bad_set)
        print(good_sample_ratio)

        mislabel_range = [(0, 0.1), (0.1, 0.3), (0.3, 0.5)]
        for low_bound, up_bound in mislabel_range:
            if low_bound <= 1 - good_sample_ratio <= up_bound:
                detect_num = int(((low_bound + up_bound) / 10) * len(train_clean_bad_set))

        print(f"######Detect {detect_num} each iteration#####")
        label_index = len(columns) - 1
        print(train_clean_bad_set.shape)
        cleaned_df = misdetect(full_data=train_clean_bad_set, clean_data=train_clean_set, bad_data=train_bad_set,
                               ground_truth=ground_truth, detect_num=detect_num, detect_iterate=5,
                               clean_data_len=int(len(train_clean_set) / 4), bad_data_len=int(len(train_bad_set) / 4),
                               early_stop=False, separate_inlier_outlier=True, weight_sampler=False)
