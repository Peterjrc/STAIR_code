import os
from load_data import load_data
import wittgenstein as lw
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def run_lof(X, y, num_outliers=560, k=60):
    clf = LocalOutlierFactor(n_neighbors=k)
    clf.fit(X)
    lof_scores = -clf.negative_outlier_factor_
    threshold = np.sort(lof_scores)[::-1][num_outliers]
    lof_predictions = np.array(lof_scores > threshold)
    lof_predictions = np.array([int(i) for i in lof_predictions])

    return lof_predictions, lof_scores


# RIPPER-K algorithm
data_dir = '../data/'
# data_list = ['Pendigits', 'PageBlock', "Shuttle", "Pima", "Mammography", "Satimage-2", "satellite",
#             "SpamBase", "optdigits", "Musk", "ALOI"]
data_list = ['optdigits']
ratio = 1


for name in data_list:
    X, y, lof_krange, N_range, knn_krange, if_range, mahalanobis_N_range = load_data(name, data_dir=data_dir)
    X,y = np.array(X), np.array(y)
    print(X.shape)
    lof_predictions, lof_scores = run_lof(X, y, k=lof_krange[7], num_outliers=int(np.sum(y)))
    outlier_idxes = np.where(lof_predictions == 1)[0]
    Outliers = X[outlier_idxes]
    idxes = np.where(lof_predictions == 0)[0]
    selected_indices = np.random.choice(np.arange(len(idxes)), size=int(len(idxes) * ratio), replace=False)
    Inliers = X[idxes[selected_indices]]
    X_new = np.concatenate([Outliers, Inliers], axis=0)
    lof_predictions_new = np.concatenate([np.ones(len(Outliers)), np.zeros(len(Inliers))], axis=0)
    X_new, y_new = X_new, lof_predictions_new

    # ripper-k

    # f1 = 0
    #
    # # while f1 < 0.72:
    # # ripper_clf = lw.RIPPER()
    # ripper_clf = lw.IREP()
    # ripper_clf.fit(X_new,y_new)
    # rule_set = ripper_clf.out_model()
    # X_test = X_new
    # y_test = y_new
    # f1 = ripper_clf.score(X_test, y_test, metrics.f1_score)
    # # accuracy = ripper_clf.score(X_test, y_test, precision_score)
    # # recall = ripper_clf.score(X_test, y_test, recall_score)
    # cond_count = ripper_clf.ruleset_.count_conds()
    # print(cond_count)
    # print('f1 score of RIPPER-K is ', f1)

# corels

import corels
import imodels
from rulefit import RuleFit
# Load the dataset
for name in data_list:
    X, y, lof_krange, N_range, knn_krange, if_range, mahalanobis_N_range = load_data(name, data_dir=data_dir)
    X, y = np.array(X), np.array(y)
    print(X.shape)
    lof_predictions, lof_scores = run_lof(X, y, k=lof_krange[7], num_outliers=int(np.sum(y)))
    outlier_idxes = np.where(lof_predictions == 1)[0]
    Outliers = X[outlier_idxes]
    idxes = np.where(lof_predictions == 0)[0]
    selected_indices = np.random.choice(np.arange(len(idxes)), size=int(len(idxes) * ratio), replace=False)
    Inliers = X[idxes[selected_indices]]
    X_new = np.concatenate([Outliers, Inliers], axis=0)
    lof_predictions_new = np.concatenate([np.ones(len(Outliers)), np.zeros(len(Inliers))], axis=0)
    df_new = pd.DataFrame(X_new, columns=[f'feature_{i}' for i in range(X_new.shape[1])])
    # print(X_new)
    # est = KBinsDiscretizer(n_bins=3, strategy='uniform', encode = 'onehot-dense')
    # X_new = est.fit_transform(X_new)
    # print(X_new)
    # c = corels.CorelsClassifier(n_iter=10000)
    # c.fit(X_new, lof_predictions_new)
    # print(c.rl())
    # # Predict on the training set
    # print(c.predict(X_new))
    # print(metrics.f1_score(lof_predictions_new, c.predict(X_new)))
    # f1 = 0
    # result = 0
    # def count_rule_list(rule_list):
    #     count = 0
    #     for string in rule_list:
    #         if "and" in string:
    #             # 如果字符串中包含 "and"，则按 "and" 分割并计算元素个数
    #             elements = string.split("and")
    #             count += len(elements)
    #         else:
    #             # 如果字符串中不包含 "and"，则长度计为 1
    #             count += 1
    #     return count
    #
    # while f1 < 0.8:
    #     clf = imodels.rule_set.rule_fit.RuleFitClassifier(max_rules = 300)
    #     clf.fit(X_new, lof_predictions_new)
    #     y_pred = clf.predict(X_new)
    #     f1 = metrics.f1_score(lof_predictions_new, y_pred)
    #     rules = clf.visualize()
    #     print(rules)
    #     print('f1 score of corels rule list is', f1)
    #
    #     rule_length = count_rule_list(rules['rule'])
    #     print('total rule length is ', rule_length)

    rf = RuleFit(max_iter=1000)
    rf.fit(X_new, lof_predictions_new, feature_names=df_new.columns)
    rules = rf.get_rules()
    rules = rules[rules.coef != 0].sort_values("support", ascending=False)

    print(len(rules))

# hics

# X_transformed = SelectKBest(f_classif, k=30).fit_transform(X_new, lof_predictions_new)
#
# for depth in range(9,25):
#     clf = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=depth)
#     # clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
#     clf.fit(X_transformed, lof_predictions_new)
#     leaf_nodes = clf.apply(X_transformed)
#     lengths = []
#     from sklearn.tree import _tree
#
#
#     def tree_to_code(tree, feature_names):
#         tree_ = tree.tree_
#         feature_name = [
#             feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
#             for i in tree_.feature
#         ]
#
#         # print("def tree({}):".format(", ".join(feature_names)))
#
#         def recurse(node, depth, attributes):
#             indent = "  " * depth
#             if tree_.feature[node] != _tree.TREE_UNDEFINED:
#                 name = feature_name[node]
#                 threshold = tree_.threshold[node]
#                 # print("{}if {} <= {}:".format(indent, name, threshold))
#                 attributes = set(attributes + [name])
#
#                 recurse(tree_.children_left[node], depth + 1, list(attributes))
#                 # print("{}else:  # if {} > {}".format(indent, name, threshold))
#
#                 recurse(tree_.children_right[node], depth + 1, list(attributes))
#             else:
#                 # print("{}return {}".format(indent, tree_.value[node]))
#                 lengths.append(len(attributes))
#
#         recurse(0, 1, [])
#
#
#     tree_to_code(clf, [str(i) for i in np.arange(X_transformed.shape[1])])
#     # print(lengths)
#     f1_score = metrics.f1_score(lof_predictions_new, clf.predict(X_transformed))
#     print(f"Depth: {depth}, f1_score: {f1_score}")
#     if f1_score > 0.8: break
# print("final F-1 score:", f1_score)
# print("whole lengths:", np.sum(lengths))
# print("lengths:", lengths)
# print("Total rule Number:", len(lengths))