import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import pandas
import torch
from get_data_tabular import *
from model import *
import operator
import math



def aggregation_feature(input_features):
    # Define the attention mechanism
    # print(input_features.shape)

    attention = nn.Linear(input_features.size(1), 1, bias=False)
    # Compute the attention weights
    weights = attention(input_features)
    weights = torch.softmax(weights, dim=0)
    # Compute the weighted sum of the input features
    weighted_sum = torch.sum(weights * input_features, dim=0)
    # Print the output
    return weighted_sum


def interactions_feature_label(neighbor_features, neighbor_labels):
    # Define the interaction layer
    interaction = nn.Linear(len(neighbor_features[0]), len(neighbor_features[0]))
    # print(label.unsqueeze(1).float())
    # Concatenate the embedded feature and label along the feature dimension

    tmp_label = np.expand_dims(neighbor_labels[0], 0)
    tmp_label = torch.tensor(tmp_label)
    x = 1
    while x < len(neighbor_features[0]):
        tmp = np.expand_dims(neighbor_labels[0], 0)
        tmp_label = torch.cat((tmp_label, torch.tensor(tmp)), 0)
        x += 1
    concatenated = torch.cat((neighbor_features[0].unsqueeze(1).float(), tmp_label.unsqueeze(1).float()), dim=1)
    # Compute the element-wise product of the embedded feature and label
    product = torch.prod(concatenated, dim=1)
    # Project the concatenated tensor using the interaction layer
    ans = interaction(product)
    ans = ans.reshape([1, len(ans)])
    # print("ans:{}".format(ans))
    # 利用interaction layer将retrieval_pool样本的特征和标签进行聚合
    # Define the embedding layer to reduce the dimensionality of the feature
    i = 1
    while i < len(neighbor_features):
        # Define the interaction layer
        interaction = nn.Linear(len(neighbor_features[i]), len(neighbor_features[i]))
        # print(label.unsqueeze(1).float())
        # Concatenate the embedded feature and label along the feature dimension
        tmp_label = np.expand_dims(neighbor_labels[i], 0)
        tmp_label = torch.tensor(tmp_label)
        x = 1
        while x < len(neighbor_features[i]):
            tmp = np.expand_dims(neighbor_labels[i], 0)
            tmp_label = torch.cat((tmp_label, torch.tensor(tmp)), 0)
            # tmp_label = torch.cat((tmp_label, torch.tensor(neighbor_labels[i])), 0)
            x += 1
        concatenated = torch.cat((neighbor_features[i].unsqueeze(1).float(), tmp_label.unsqueeze(1).float()), dim=1)

        # Compute the element-wise product of the embedded feature and label
        product = torch.prod(concatenated, dim=1)

        # Project the concatenated tensor using the interaction layer
        projected = interaction(product)

        ans = torch.cat((ans, projected.reshape([1, len(projected)])), 0)
        # Print the output
        i += 1
    return aggregation_feature(ans)
    # print(projected.squeeze(0))


def misdetect(
        dataset, detect_num, detect_iterate, clean_pool_len, bad_pool_len, confidence_threshold, train_clean_bad_set, train_clean_dataset, train_bad_dataset, train_clean_bad_set_ground_truth):
    total_len = len(train_clean_bad_set)
    clean_len = len(train_clean_dataset)
    bad_len = len(train_bad_dataset)

    new_train_clean_bad_set = copy.deepcopy(train_clean_bad_set)
    train_clean_bad_set_copy = copy.deepcopy(train_clean_bad_set)
    ground_truth = copy.deepcopy(train_clean_bad_set_ground_truth)

    train_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=1, shuffle=False)

    epoch = 2

    # 总共的correct个数
    total_correct_num = 0
    total_detect_num = 0
    final_dirty_set = []
    clean_idx = []

    num_clean0 = 0
    num_bad1 = 0
    model = Model(len(columns[dataset]) - 1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)

    retrieval_pool = []
    start_time = time.time()

    # 800 / 50 = 16
    for times in range(confidence_threshold):
        # if (times * detect_num) >= 0.3 * len(train_bad_dataset):
        #     print("------------------------------------kaishi----------------------------------------")
        #     model = Model()
        #     model = model.to(device)
        #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #     loss_function = nn.CrossEntropyLoss()
        #     loss_function = loss_function.to(device)
        #     epoch = 5
        # if times >= 1:
        #     print("[][][][][][][][][]")
        #     epoch = 3
        # model = Model()
        # model = model.to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # loss_function = nn.CrossEntropyLoss()
        # loss_function = loss_function.to(device)

        model.train()
        early_loss = np.zeros(total_len, dtype=np.float64)
        for i in range(epoch):
            for data in train_loader:
                # GPU加速
                train_feature, train_label = data
                train_feature = train_feature.to(device)
                train_label = train_label.to(device)
                optimizer.zero_grad()
                train_label_predict = model(train_feature)

                # GPU加速
                train_label_predict = train_label_predict.to(device)
                train_loss = loss_function(train_label_predict, train_label)
                train_loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                num = 0
                for data in test_loader:
                    test_feature, test_label = data
                    test_feature = test_feature.to(device)
                    test_label = test_label.to(device)
                    test_label_predict = model(test_feature)
                    test_label_predict = test_label_predict.to(device)
                    loss = loss_function(test_label_predict, test_label)
                    early_loss[num] += loss
                    num += 1

        loss_clean = sorted(enumerate(list(early_loss)),
                            key=lambda x: x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
        clean_pool_idx = [x[0] for x in loss_clean]  # 获取排序好后b坐标,下标在第0位

        model_clean = Model(len(columns[dataset]) - 1)
        model_clean = model_clean.to(device)
        optimizer_clean = torch.optim.Adam(model_clean.parameters(), lr=0.001)
        loss_function_clean = nn.CrossEntropyLoss()
        loss_function_clean = loss_function_clean.to(device)

        model_clean.train()
        clean_pool = []
        num_add = 0
        indeX = 0
        while num_add < clean_pool_len:
        # for indeX in range(clean_pool_len):

        # extract clean instances
            if clean_pool_idx[indeX] not in clean_idx:
                clean_idx.append(clean_pool_idx[indeX])
                tmp_pca = copy.deepcopy(train_clean_bad_set[clean_pool_idx[indeX]][0].tolist())
                tmp_pca.append(train_clean_bad_set[clean_pool_idx[indeX]][1])
                tmp_pca = [torch.tensor(tmp_pca), 0]
                # tmp_pca.append(0)
                retrieval_pool.append(tmp_pca)
                num_clean0 += 1
                num_add += 1
            indeX += 1

        for indeX in range(clean_pool_len):
            clean_pool.append(train_clean_bad_set[clean_pool_idx[indeX]])
        train_loader_clean = DataLoader(dataset=MyDataSet(clean_pool), batch_size=64, shuffle=True)

        #***************************************
        # if times < confidence_threshold:
        #     for i in clean_pool:
        #         i.append(0)
        #         retrieval_pool.append(i)
        # from sklearn.decomposition import PCA
        # # from sklearn.preprocessing import StandardScaler
        # pca = PCA(n_components=1)
        # for i in clean_pool:
        #     # tmp_pca = pca.fit_transform(i)
        #     tmp_pca = copy.deepcopy(i[0].tolist())
        #     tmp_pca.append(i[1])
        #     tmp_pca = [torch.tensor(tmp_pca), 0]
        #     # tmp_pca.append(0)
        #     retrieval_pool.append(tmp_pca)
        # ***************************************

        # 训练干净数据集到收敛
        for epo in range(40):
            for data in train_loader_clean:
                # GPU加速
                train_feature_clean, train_label_clean = data
                train_feature_clean = train_feature_clean.to(device)
                train_label_clean = train_label_clean.to(device)
                optimizer_clean.zero_grad()
                train_label_predict_clean = model_clean(train_feature_clean)

                # GPU加速
                train_label_predict_clean = train_label_predict_clean.to(device)
                train_loss_clean = loss_function_clean(train_label_predict_clean, train_label_clean)
                train_loss_clean.backward()
                optimizer_clean.step()
            model_clean.eval()
            with torch.no_grad():
                test_accuracy_total = 0
                for data in train_loader_clean:
                    test_feature_total, test_label_total = data
                    test_feature_total = test_feature_total.to(device)
                    test_label_total = test_label_total.to(device)
                    test_label_predict_total = model(test_feature_total)
                    test_accuracy_num_total = (test_label_predict_total.argmax(1) == test_label_total).sum()
                    test_accuracy_total += test_accuracy_num_total
                if (test_accuracy_total / clean_pool_len) >= 0.9:
                    break
        loss_bad = sorted(enumerate(list(early_loss)),
                          key=lambda x: -x[1])  # x[1]是因为在enumerate(early_loss)中，early_loss数值在第1位
        bad_pool_idx = [x[0] for x in loss_bad]  # 获取排序好后b坐标,下标在第0位

        # ***************************************
        bad_pool = []
        for indeX in range(bad_pool_len):
            bad_pool.append(train_clean_bad_set[bad_pool_idx[indeX]])
        # if times < confidence_threshold:
        #     for i in bad_pool:
        #         i.append(1)
        #         retrieval_pool.append(i)
        from sklearn.decomposition import PCA
        # from sklearn.preprocessing import StandardScaler
        pca = PCA(n_components=1)
        # extract bad data
        for i in bad_pool:
            # tmp_pca = pca.fit_transform(i)
            tmp_pca = copy.deepcopy(i[0].tolist())
            tmp_pca.append(i[1])
            tmp_pca = [torch.tensor(tmp_pca), 1]
            # tmp_pca.append(0)
            retrieval_pool.append(tmp_pca)
            num_bad1 += 1
        # ***************************************

        bad_pool_idx = bad_pool_idx[:bad_pool_len]
        influence_bad = []
        model_clean.eval()
        for indeX in bad_pool_idx:
            bad_test = [train_clean_bad_set[indeX]]
            test_loader_bad = DataLoader(dataset=MyDataSet(bad_test), batch_size=1, shuffle=False)
            for data in test_loader_bad:
                train_feature_bad, train_label_bad = data
                train_feature_bad = train_feature_bad.to(device)
                train_label_bad = train_label_bad.to(device)
                train_label_predict_bad = model_clean(train_feature_bad)

                # GPU加速
                train_label_predict_bad = train_label_predict_bad.to(device)
                train_loss_bad = loss_function_clean(train_label_predict_bad, train_label_bad)

                train_loss_bad_gradient = torch.autograd.grad(train_loss_bad, model_clean.parameters(), allow_unused=True)

                grad = 0.0
                for item in train_loss_bad_gradient:
                    if item == None:
                        continue
                    item = torch.norm(item)
                    grad += item

                tmp = [indeX, grad]
                influence_bad.append(tmp)

        influence_bad_sorted = sorted(influence_bad, key=lambda x: -x[1])
        influence_bad_idx = [x[0] for x in influence_bad_sorted]  # 获取排序好后b坐标,下标在第0位

        correct_num = 0
        true_bad_detected_idx = []
        detect_idx_50 = []

        # 每轮detect500个
        for i in range(detect_num):
            total_detect_num += 1
            detect_idx_50.append(influence_bad_idx[i])
            if (total_len - 1) >= influence_bad_idx[i] >= clean_len:
                correct_num += 1
                true_bad_detected_idx.append(influence_bad_idx[i])

        print("loss最高的脏数据占比为:{}".format(correct_num / detect_num))
        total_correct_num += correct_num

        pre = total_correct_num / total_detect_num
        rec = total_correct_num / len(train_bad_dataset)
        # 计算总的精度
        print("第{}轮的precision{},recall{},f1{}".format(times + 1, pre, rec, 2 * pre * rec / (pre + rec)))

        ground_truth_tmp = []
        new_train_clean_bad_set = []
        for i in range(total_len):
            if i not in detect_idx_50:
                new_train_clean_bad_set.append(train_clean_bad_set_copy[i])
                ground_truth_tmp.append(ground_truth[i])
            else:
                final_dirty_set.append(ground_truth[i])
        # for i in range(total_len):
        #     if i not in detect_idx_50:
        #         train_clean_bad_set_tmp.append(train_clean_bad_set_copy[i])
        train_clean_bad_set_copy = copy.deepcopy(new_train_clean_bad_set)
        ground_truth = copy.deepcopy(ground_truth_tmp)

        total_len = len(new_train_clean_bad_set)
        bad_len = bad_len - correct_num
        clean_len = total_len - bad_len
        print(clean_len, bad_len, total_len)

        train_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=128, shuffle=True)
        test_loader = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=1, shuffle=False)

    print("训练集中，干净：{}，脏：{}".format(num_clean0, num_bad1))
    end_time = time.time()
    print("GPU time:{}".format(end_time - start_time))
    # total_len += len(outlier_verify)
    # ***************************************
    retrieval_pool_test = []
    correct_num = 0
    from sklearn.decomposition import PCA
    # from sklearn.preprocessing import StandardScaler
    pca = PCA(n_components=1)
    for i in new_train_clean_bad_set:
        tmp_pca = copy.deepcopy(i[0].tolist())
        tmp_pca.append(i[1])
        tmp_pca = [torch.tensor(tmp_pca), 2]
        # tmp_pca.append(0)
        retrieval_pool_test.append(tmp_pca)
    # ***************************************


    # 收集KNN训练数据集***************************************
    train_pool = copy.deepcopy(retrieval_pool + retrieval_pool_test)
    features = []
    labels = []
    for item in train_pool:
        features.append(item[0])
        labels.append(item[1])
    features = pandas.DataFrame(features)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree').fit(features)

    train_retrieval_pool = []
    for i in range(len(retrieval_pool)):
        distance, index = nbrs.kneighbors(retrieval_pool[i][0].reshape(1, -1), 10)
        neighbors = []
        neighbors_labels = []
        for x in np.array(index.tolist()[0]):
            neighbors.append(train_pool[x][0])
            neighbors_labels.append(train_pool[x][1])
        # neighbors_labels = torch.tensor(neighbors_labels)
        input_features = neighbors[0].reshape([1, len(neighbors[0])])
        id = 1
        while id < len(neighbors):
            input_features = torch.cat([input_features, neighbors[id].reshape([1, len(neighbors[id])])], 0)
            id += 1
        # 利用attention将retrieval_pool样本的特征进行聚合
        weighted_sum = aggregation_feature(input_features)

        # 利用interaction layer将retrieval_pool样本的特征和标签进行聚合
        aggregated = interactions_feature_label(neighbors, neighbors_labels)

        train_feature_retrieval = torch.cat((retrieval_pool[i][0], weighted_sum, aggregated), 0)
        train_label_retrieval = retrieval_pool[i][1]
        tmp_retrieval = [train_feature_retrieval.detach(), train_label_retrieval]
        # print(tmp_retrieval)

        train_retrieval_pool.append(tmp_retrieval)
    # ***************************************

    # 收集KNN测试数据集***************************************
    test_retrieval_pool = []
    for i in range(len(retrieval_pool_test)):
        # 对于retrieval pool里的每一个instance找出它的k个knn instance并聚合他们的features
        distance, index = nbrs.kneighbors(retrieval_pool_test[i][0].reshape(1, -1), 10)
        neighbors = []
        neighbors_labels = []
        for x in np.array(index.tolist()[0]):
            neighbors.append(train_pool[x][0])
            neighbors_labels.append(train_pool[x][1])
        # neighbors_labels = torch.tensor(neighbors_labels)
        input_features = neighbors[0].reshape([1, len(neighbors[0])])
        id = 1
        while id < len(neighbors):
            input_features = torch.cat([input_features, neighbors[id].reshape([1, len(neighbors[id])])], 0)
            id += 1
        # 利用attention将retrieval_pool样本的特征进行聚合
        weighted_sum = aggregation_feature(input_features)
        # 利用interaction layer将retrieval_pool样本的特征和标签进行聚合
        aggregated = interactions_feature_label(neighbors, neighbors_labels)

        test_feature_retrieval = torch.cat((retrieval_pool_test[i][0], weighted_sum, aggregated), 0)
        tmp_retrieval = test_feature_retrieval.detach()
        test_retrieval_pool.append(tmp_retrieval)
    # ***************************************

    # 训练retrieval_pool---------------------------------
    train_loader_retrieval = DataLoader(dataset=MyDataSet(train_retrieval_pool), batch_size=128, shuffle=True)
    # test_loader_retrieval = DataLoader(dataset=MyDataSet(new_train_clean_bad_set), batch_size=1, shuffle=False)
    model_retrieval = Model_retrieval(3*len(columns[dataset]))
    model_retrieval = model_retrieval.to(device)
    optimizer_retrieval = torch.optim.Adam(model_retrieval.parameters(), lr=0.001)
    loss_function_retrieval = nn.MSELoss()
    loss_function_retrieval = loss_function_retrieval.to(device)
    model_retrieval.train()
    for i in range(20):
        for data in train_loader_retrieval:
            # GPU加速
            train_feature, train_label = data
            train_feature = train_feature.to(device)
            train_label = train_label.to(device)
            optimizer_retrieval.zero_grad()
            train_label_predict = model_retrieval(train_feature)

            # GPU加速
            train_label_predict = train_label_predict.to(device)
            train_loss = loss_function_retrieval(train_label_predict.squeeze().double(), train_label.double())
            train_loss.backward()
            optimizer_retrieval.step()
    # 训练retrieval_pool结束，测试retrieval_pool开始---------------------------------
    test_loader_retrieval = DataLoader(dataset=MyDataSet(test_retrieval_pool), batch_size=1, shuffle=False)
    model_retrieval.eval()
    with torch.no_grad():
        num = 0
        hh = 0
        for data in test_loader_retrieval:
            test_feature = data
            test_feature = test_feature.to(device)
            test_label_predict = model_retrieval(test_feature)
            if num not in clean_idx:
                hh += 1
            if test_label_predict > 0.50:
                if num not in clean_idx:
                    print(num, ":", test_label_predict, len(clean_idx))
                    total_detect_num += 1
                    if (total_len - 1) >= num >= clean_len:
                        correct_num += 1
            num += 1
    # 测试retrieval_pool结束***************************************
    print(len(clean_idx), num, hh)
    total_correct_num += correct_num

        # 计算总的精度
    print("第{}轮的precision{},recall{}".format(confidence_threshold + 1, total_correct_num / total_detect_num,
                                               total_correct_num / len(train_bad_dataset)))


    precision = total_correct_num / total_detect_num
    recall = total_correct_num / len(train_bad_dataset)
    f1 = 2 * (precision * recall) / (precision + recall)
    # print("detect_iterate * detect_num:{}".format(detect_iterate * detect_num))
    print("detect precision：{}".format(precision))
    print("detect recall：{}".format(recall))
    print("detect f1 score：{}".format(f1))

