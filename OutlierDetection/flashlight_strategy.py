import pandas as pd
from lca import lca, is_approx_equal
from information_gain import information_gain
from load_data import load_data
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import os

def generate_lca_set(sample, data):
    """生成LCA集合"""
    lca_set = set()
    for s in sample:
        for d in data:
            lca_set.add(lca(s, d))
    return lca_set


def calculate_statistics(pattern, data, positive_index):
    """计算给定模式的统计信息"""
    support = sum(1 for row in data if all(
        # 判断是否相等，数值就判断是否近似相等
        attr == '*' or attr == value or (isinstance(attr, (int, float)) and is_approx_equal(attr, value))
        for attr, value in zip(pattern, row[:-1])
    ))
    # print(f"Support for pattern {pattern}: {support}")  # 打印支持度

    positive_count = sum(1 for row in data if all(
        attr == '*' or attr == value or (isinstance(attr, (int, float)) and is_approx_equal(attr, value))
        for attr, value in zip(pattern, row[:-1])
    ) and row[positive_index] == 1)
    # print(f"Positive count for pattern {pattern}: {positive_count}")  # 打印正例计数

    positive_rate = positive_count / support if support else 0
    # print(f"Positive rate for pattern {pattern}: {positive_rate}")  # 打印正例比率

    return support, positive_rate

def calculate_precision_recall_f1(explanation_table, data, positive_index):
    # 初始化记录每个数据点符合的模式的集合
    pattern_matches = [set() for _ in data]

    # 遍历解释表中的每个模式
    for pattern_index, pattern in enumerate(explanation_table):
        for i, row in enumerate(data):
            if all(
                attr == '*' or attr == value or (isinstance(attr, (int, float)) and is_approx_equal(attr, value))
                for attr, value in zip(pattern['Pattern'], row[:-1])
            ):
                pattern_matches[i].add(pattern_index)

    # 初始化计数器
    true_positives = 0
    false_positives = 0
    counted = set()  # 用于记录已经计数的数据点

    # 遍历每个数据点的匹配模式集合
    for i, matches in enumerate(pattern_matches):
        if matches:  # 如果有匹配的模式
            if i not in counted:  # 检查是否已计数
                counted.add(i)  # 标记为已计数
                if data[i][positive_index] == 1:
                    true_positives += 1
                else:
                    false_positives += 1

    # 计算其他指标
    total_positives = sum(1 for row in data if row[positive_index] == 1)
    false_negatives = total_positives - true_positives

    if(false_negatives < 0):
        print("Error")
        exit(0)

    # 计算精确度、召回率和 F1 分数
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1



def flashlight_strategy(sample, data, positive_index, max_patterns=None, min_info_gain=None, min_f1=0.8):
    lca_set = generate_lca_set(sample, data)
    explanation_table = []
    lca_set = list(lca_set)

    # 直到LCA集合为空，或者已经达到了最大模式数量
    while lca_set and (max_patterns is None or len(explanation_table) < max_patterns):
        max_info_gain = float('-inf')
        best_pattern = None

        # num = 0
        # 遍历LCA集合以找到具有最大信息增益的模式

        for pattern in lca_set[:int(0.1*len(lca_set))]:
            current_info_gain = information_gain(pattern, data, positive_index)
            # print(num)
            # num += 1
            if current_info_gain > max_info_gain:
                max_info_gain = current_info_gain
                best_pattern = pattern

        # 检查是否找到了有效模式，且是否超过了最小信息增益阈值
        if best_pattern is None or (min_info_gain is not None and max_info_gain < min_info_gain):
            break

        # 添加最佳模式到解释表
        lca_set.remove(best_pattern)
        support, positive_rate = calculate_statistics(best_pattern, data, positive_index)
        explanation_table.append({
            'Pattern': best_pattern,
            'Support': support,
            'Positive Rate': positive_rate,
            'Information Gain': max_info_gain
        })

        # 计算并检查 F1 分数
        _, _, f1 = calculate_precision_recall_f1(explanation_table, data, positive_index)
        print(f1, len(explanation_table))
        if min_f1 is not None and f1 >= min_f1:
            break

    explanation_table_df = pd.DataFrame(explanation_table)
    return explanation_table_df



def run_lof(X, y, num_outliers=560, k=60):
    clf = LocalOutlierFactor(n_neighbors=k)
    clf.fit(X)
    lof_scores = -clf.negative_outlier_factor_
    threshold = np.sort(lof_scores)[::-1][num_outliers]
    lof_predictions = np.array(lof_scores > threshold)
    lof_predictions = np.array([int(i) for i in lof_predictions])

    return lof_predictions, lof_scores


if __name__ == '__main__':
    data_dir = '../data/'
    data_list = ['skin']
    ratio = 0.02

    for name in data_list:
        X, y, lof_krange, N_range, knn_krange, if_range, mahalanobis_N_range = load_data(name, data_dir=data_dir)
        X, y = np.array(X), np.array(y)
        print(X.shape)
        lof_predictions, lof_scores = run_lof(X, y, k=lof_krange[7], num_outliers=int(np.sum(y)))
        outlier_idxes = np.where(lof_predictions == 1)[0]
        Outliers = X[outlier_idxes]
        idxes = np.where(lof_predictions == 0)[0]
        selected_indices = np.random.choice(np.arange(len(idxes)), size=int(len(idxes) * ratio), replace=False)
        Inliers = X[idxes][selected_indices]
        X_new = np.concatenate([Outliers, Inliers], axis=0)
        lof_predictions_new = np.concatenate([np.ones(len(Outliers)), np.zeros(len(Inliers))], axis=0)

        sample = Inliers[:1]
        data = np.concatenate([X_new, lof_predictions_new.reshape(-1, 1)], axis=1)

    df = flashlight_strategy(sample, data, -1, min_f1=0.8)
    print(df)