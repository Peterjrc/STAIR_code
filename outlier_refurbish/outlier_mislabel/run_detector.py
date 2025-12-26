import os
import numpy as np
from load_data import load_data
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from pyod.models.iforest import IForest


def run_detector(args):

    X, y, lof_krange, N_range, knn_krange, if_range, mahalanobis_N_range = load_data(args)
    X = np.array(X)
    y = np.array(y)
    y = y.reshape(-1, 1)
    X_df = pd.DataFrame(X, columns=[f'attr{i}' for i in range(1, X.shape[1] + 1)])
    outlier_ratio = np.mean(y)

    def run_lof(X, y, filename, num_outliers=560, k=60):
        clf = LocalOutlierFactor(n_neighbors=lof_krange[7])

        clf.fit(X)
        lof_scores = -clf.negative_outlier_factor_
        threshold = np.sort(lof_scores)[::-1][num_outliers]
        lof_predictions = np.array(lof_scores > threshold)
        lof_predictions = np.array([int(i) for i in lof_predictions])
        f1_score = metrics.f1_score(y, lof_predictions)
        print(f"F-1 score of LOF outlier detection for {filename}:", f1_score)
        recall_score = metrics.recall_score(y, lof_predictions)
        precision_score = metrics.precision_score(y, lof_predictions)
        return lof_predictions, lof_scores, f1_score, recall_score, precision_score

    lof_predictions, lof_scores, f1_score, recall_score, precision_score = run_lof(X, y, filename=args.dataset, k=lof_krange[7], num_outliers=int(np.sum(y)))

    # def run_if_(X, y, outlier_ratio):
    #     if_model = IForest(contamination=outlier_ratio)
    #     if_model.fit(X)
    #     pred_labels = if_model.predict(X)
    #     outlierness_score = if_model.decision_function(X)
    #     print(f"F1 score of Isolation Forest: {metrics.f1_score(y, pred_labels)}")
    #     return pred_labels, outlierness_score
    #
    # lof_predictions, lof_scores = run_if_(X, y, outlier_ratio)

    lof_predictions = pd.DataFrame(lof_predictions, columns=['label'])
    gt_labels = pd.DataFrame(y, columns=['label'])
    df_lof = pd.concat((X_df, lof_predictions), axis=1)
    df_real = pd.concat((X_df, gt_labels), axis=1)
    print('nan number:', df_lof.isnull().sum().sum())
    lof_scores = MinMaxScaler().fit_transform(lof_scores.reshape(-1, 1))
    # use scaler to generate soft labels
    soft_label = np.concatenate([1 - lof_scores, lof_scores], axis=1)

    good_sample_ratio = np.sum(df_real['label'] == df_lof['label']) / len(df_real)
    noisy_rate = 1 - good_sample_ratio
    mislabel_num = np.sum(df_real['label'] != df_lof['label'])
    outlier_num = np.sum(df_lof['label'] == 1)
    inlier_num = np.sum(df_lof['label'] == 0)
    outlier_ratio = df_lof['label'].mean()
    outlier_noisy_rate = (mislabel_num / 2) / outlier_num
    inlier_noisy_rate = (mislabel_num / 2) / inlier_num
    print('outlier noisy rate:', outlier_noisy_rate, 'inlier noisy rate:', inlier_noisy_rate)
    print(f"Total mislabeled number {mislabel_num}")
    print(df_lof.head())
    X = X_df.values
    y_real = df_real['label'].values.reshape(-1,1)
    y_noisy = df_lof['label'].values.reshape(-1,1)
    dim = X.shape[1]

    return X, y_real, y_noisy, soft_label, dim, outlier_ratio



