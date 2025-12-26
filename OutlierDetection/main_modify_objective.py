import math
import sys

import numpy as np
from sklearn import metrics
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
import pandas as pd
from load_data import load_data, load_misdetect_prediction


def cal_f1score(stats_list, return_detail=False):
    if len(stats_list) == 0: # if the rule list is empty
        if return_detail:
            return tuple([0] * 4)
        else:
            return 0

    all_statistics = [[stat.label * stat.num * stat.acc,
                       stat.label * stat.num * (1 - stat.acc),
                       (1 - stat.label) * stat.num * (1 - stat.acc)] for stat in stats_list]
    TP = np.sum(np.array(all_statistics)[:, 0])
    FP = np.sum(np.array(all_statistics)[:, 1])
    FN = np.sum(np.array(all_statistics)[:, 2])
    if return_detail:
        return TP / (TP + (FP + FN) / 2 + 1e-5), TP, FP, FN
    else:
        return TP / (TP + (FP + FN) / 2 + 1e-5)


def cal_recall(stats_list, return_detail=False):
    if len(stats_list) == 0: # if the rule list is empty
        if return_detail:
            return tuple([0] * 4)
        else:
            return 0

    all_statistics = [[stat.label * stat.num * stat.acc,
                       stat.label * stat.num * (1 - stat.acc),
                       (1 - stat.label) * stat.num * (1 - stat.acc)] for stat in stats_list]
    TP = np.sum(np.array(all_statistics)[:, 0])
    FP = np.sum(np.array(all_statistics)[:, 1])
    FN = np.sum(np.array(all_statistics)[:, 2])
    if return_detail:
        return TP / (TP + FN + 1e-5), TP, FP, FN
    else:
        return TP / (TP + FN + 1e-5)


def cal_precision(stats_list, return_detail=False):
    if len(stats_list) == 0: # if the rule list is empty
        if return_detail:
            return tuple([0] * 4)
        else:
            return 0

    all_statistics = [[stat.label * stat.num * stat.acc,
                       stat.label * stat.num * (1 - stat.acc),
                       (1 - stat.label) * stat.num * (1 - stat.acc)] for stat in stats_list]
    TP = np.sum(np.array(all_statistics)[:, 0])
    FP = np.sum(np.array(all_statistics)[:, 1])
    if return_detail:
        return TP / (TP + FP + 1e-5), TP, FP
    else:
        return TP / (TP + FP + 1e-5)


def entropy(rule):
    p = rule.acc
    ent = - p * np.log(p + 1e-8) - (1 - p) * np.log(1 - p + 1e-8)
    return 1 - ent



def cal_list_entropy(stats_list, return_detail=False):
    if len(stats_list) == 0:
        if return_detail:
            return tuple([0] * 2)
        else:
            return 0

    entropy_list = np.array([1 - entropy(rule) for rule in stats_list]) # entropy for each rule
    lengths = np.array([rule.num for rule in stats_list]) # number of instances fall into rule i

    total_length = np.sum(lengths) # total length of the dataset
    avg_entropy = np.sum(entropy_list * lengths / total_length) # weighted average entropy
    if return_detail:
        return avg_entropy, total_length
    else:
        return avg_entropy


class Node(object):
    def __init__(self):
        self.value = None
        self.decision = None
        self.lchild = None
        self.rchild = None
        self.is_leaf = True
        self.split_num = None # splitting value in the feature col
        # self.attribute = None

    def set_rule(self, rule):
        self.rule = rule

    def predict_single(self, x): # for each single instance
        if not self.is_leaf: # if the node is not leaf node
            if x[self.rule.split_attribute] <= self.rule.split_num:
            # self.rule.split_attribute: the attribute being splitted
            # self.rule.split_num: the splitting threshold of the attribute
                return self.lchild.predict_single(x)
            else:
                return self.rchild.predict_single(x)
        else: # if the node is leaf node, then output the prediction label
            return self.rule.label

    def predict(self, X):
        # make prediction for each instance in the dataset
        return np.array([self.predict_single(x) for x in X])


def findEntropy(data, rows):

    # calculates the entropy for a subset of rows in a given dataset,
    # specifically focusing on binary classification with labels 'Yes' and 'No'.
    # calculate the entropy of the current dataset (before splitting).
    # This tells you how impure the dataset is with respect to the target labels.
    yes = 0 # label
    no = 0 # label
    ans = -1
    idx = len(data[0]) - 1 # index storing the label
    entropy = 0
    for i in rows:
        if data[i][idx] == 'Yes':
            yes = yes + 1 # count the number of yes
        else:
            no = no + 1 # count the number of no

    x = yes / (yes + no) # proportion of yes label
    y = no / (yes + no) # proportion of no label
    if x != 0 and y != 0:
        entropy = -1 * (x * math.log2(x) + y * math.log2(y)) # calculate entropy
    if x == 1:
        ans = 1 # if all labels in rows are 1 then the entropy is 0
    if y == 1:
        ans = 0 # if all labels in rows are 1 then the entropy is 0
    return entropy, ans


def cal_split_entropy(X, y, cur_rule, col, val):
    # Information Gain = Entropy(before split) - Weighted Average Entropy(after split)
    # calculate the entropy after splitting
    # col: feature column
    # val: the threshold value in col used to split data
    idxes_1 = cur_rule.idxes[np.where(X[cur_rule.idxes, col] <= val)]
    # idxes_1: Indices of data points where the value in column col is less than or equal to val
    idxes_2 = cur_rule.idxes[np.where(X[cur_rule.idxes, col] > val)]
    # idxes_2: Indices of data points where the value in column col is greater than val

    tmp_y_1 = y[idxes_1]
    tmp_y_2 = y[idxes_2]

    label_1 = int(np.sum(tmp_y_1) > len(tmp_y_1) / 2) # compute the majority labels of subset 1
    label_2 = int(np.sum(tmp_y_2) > len(tmp_y_2) / 2) # compute the majority labels of subset 2
    # len(tmp_y_1) / 2 gives the threshold of majority: 50%
    # This is a comparison that checks whether the number of 1s in tmp_y_1 (i.e., np.sum(tmp_y_1))
    # is greater than half the total number of elements in tmp_y_1

    assert len(tmp_y_1) > 0
    assert len(tmp_y_2) > 0

    acc_1 = np.sum(tmp_y_1 == label_1) / len(tmp_y_1) # proportion of majority label in subset 1
    acc_2 = np.sum(tmp_y_2 == label_2) / len(tmp_y_2) # proportion of majority label in subset 2

    # compute the weighted average entropy after splitting
    new_avg_entropy = len(tmp_y_1) * (1 - (-acc_1 * np.log(acc_1 + 1e-5)
                                           - (1 - acc_1) * np.log(1 - acc_1 + 1e-5))) + len(tmp_y_2) * (
                              1 - (-acc_2 * np.log(acc_2 + 1e-5)
                                   - (1 - acc_2) * np.log(1 - acc_2 + 1e-5)))
    return new_avg_entropy


def cal_split_miss_outliers(X, y, cur_rule, col, val):
    idxes_1 = cur_rule.idxes[np.where(X[cur_rule.idxes, col] <= val)]
    # idxes_1: Indices of data points where the value in column col is less than or equal to val
    idxes_2 = cur_rule.idxes[np.where(X[cur_rule.idxes, col] > val)]
    # idxes_2: Indices of data points where the value in column col is greater than val

    tmp_y_1 = y[idxes_1]
    tmp_y_2 = y[idxes_2]

    label_1 = int(np.sum(tmp_y_1) > len(tmp_y_1) / 2)  # compute the majority labels of subset 1
    label_2 = int(np.sum(tmp_y_2) > len(tmp_y_2) / 2)  # compute the majority labels of subset 2

    assert len(tmp_y_1) > 0
    assert len(tmp_y_2) > 0

    miss_outliers_1 = np.sum((tmp_y_1 == 1) & (np.full(len(tmp_y_1), label_1) == 0))
    miss_outliers_2 = np.sum((tmp_y_2 == 1) & (np.full(len(tmp_y_2), label_2) == 0))

    new_miss_outliers = miss_outliers_1 + miss_outliers_2

    return new_miss_outliers



def findMaxGain(X, y, stats_list, rows, columns, cur_rule):
    maxGain = 0
    best_split_num = None
    rows_smaller = None
    rows_larger = None
    best_attribute = None

    if len(np.unique(y[rows])) == 1:
        # checks if all the labels in the current subset (y[rows]) are the same (i.e., if the data is pure).
        return maxGain, best_attribute, best_split_num, rows_smaller, rows_larger

    # len_of_rules = np.sum([len(stat.attributes) for stat in stats_list])

    avg_entropy0, total_length0 = cal_list_entropy(stats_list, return_detail=True) # compute current average entropy based on stats_list
    avg_entropy = (avg_entropy0 * total_length0 + entropy(cur_rule) * cur_rule.num) / (cur_rule.num + total_length0)
    cur_metric = avg_entropy

    best_entropy = -1000
    for col in columns: # for every feature

        for val in np.sort(np.unique(X[cur_rule.idxes, col]))[:-1]: # for each value of each feature (column)

            new_avg_entropy = cal_split_entropy(X, y, cur_rule, col, val)

            if new_avg_entropy > best_entropy:
                best_attribute = col
                maxGain = new_avg_entropy - cur_metric # compute information gain
                best_split_num = val
                rows_smaller = cur_rule.idxes[np.where(X[cur_rule.idxes, col] <= val)] # indces of rows fall into left child node
                rows_larger = cur_rule.idxes[np.where(X[cur_rule.idxes, col] > val)]
                best_entropy = new_avg_entropy

    if maxGain < 0:
        print("maxgain < 0")

    return maxGain, best_attribute, best_split_num, rows_smaller, rows_larger


def buildTree(X, predictions, stats, cur_rule, stats_list, rows, columns):
    '''
    start building the decision tree for data X[rows][columns]
    params:
    X: all data
    predictions: lof predictions
    rows: rows we need to deal with now
    columns: columns we need to deal with now (usually full data columns)
    attributes: array of numbers, the columns/attributes that has been used before
    stats_list: the rules outside of current data X
    '''

    # first need to find the attribute, the split number, the points that are larger or smaller than the split number;
    # to maximize the objective

    maxGain, attribute, split_num, rows_smaller, rows_larger = findMaxGain(X, predictions, stats_list, rows, columns,
                                                                           cur_rule)

    if maxGain == 0:
        # it means we cannot split the current rule further, reaches the leaf node
        root = Node() # initiate the root node
        root.is_leaf = True
        root.value = int(np.sum(predictions[rows]) > len(rows) / 2)
        return root

    print("maxGain:", maxGain, "idxes_smaller:", len(rows_smaller), "idxes_larger", len(rows_larger), "rule length",
          len(list(set(cur_rule.attributes + [attribute]))), "split_num:", split_num)

    # split the current rule, the same as splitting the current data X[rows]
    # rows_smaller = idxes_smaller
    # rows_larger = idxes_larger
    # split the samples based on current rule into two leaves
    label_1 = int(np.sum(predictions[rows_smaller]) > len(rows_smaller) / 2)
    label_2 = int(np.sum(predictions[rows_larger]) > len(rows_larger) / 2)
    acc_1 = np.sum(predictions[rows_smaller] == label_1) / len(rows_smaller) # compute proportion of label 1 after splitting
    acc_2 = np.sum(predictions[rows_larger] == label_2) / len(rows_larger)

    # form two new rules
    rule_1 = stats(len(rows_smaller), acc_1, label_1, rows_smaller, list(set(cur_rule.attributes + [attribute])))
    rule_2 = stats(len(rows_larger), acc_2, label_2, rows_larger, list(set(cur_rule.attributes + [attribute])))

    print("best attribute:", attribute, "label 1:", label_1, "num 1:", len(rows_smaller), "label 2:", label_2, "num 2:",
          len(rows_larger))

    root = Node() # the root node of the whole tree
    root.is_leaf = False
    root.attribute = attribute
    root.split_num = split_num
    # root.childs = []

    # continue to build left child node recursively ----> find maximum gain
    root.lchild = buildTree(X, predictions, stats,
                            cur_rule=rule_1,
                            stats_list=stats_list + [rule_2],
                            rows=rows_smaller, columns=columns)
    if root.lchild.is_leaf:
        root.lchild.rule = rule_1
    # continue to build right child node
    root.rchild = buildTree(X, predictions, stats,
                            cur_rule=rule_2,
                            stats_list=stats_list + [rule_1],
                            rows=rows_larger, columns=columns)
    if root.rchild.is_leaf:
        root.rchild.rule = rule_2

    return root

# note stats_list is used to contain rules
# each rule stats( ) has a list of attributes in it


def update_attributes_and_intervals(cur_attributes, cur_intervals, attribute, value, mode='left'):
    # cur_attributes: A list of attributes (features) that have already been used for splitting
    # cur_intervals: A list of intervals (ranges) corresponding to each attribute in cur_attributes
    # attribute: The new attribute (feature) that is being used for the split at the current step
    # value: The value of the attribute that is used for the split. This is the threshold value for partitioning the data
    # mode: Indicates whether the split is for the left child or the right child of the current node

    if attribute in cur_attributes: # if the new attribute is already in the cur_attributes list
        idx = cur_attributes.index(attribute)
        assert cur_intervals[idx][1] >= value and cur_intervals[idx][0] <= value

    else: # add new attribute into cur_attribute list
        idx = -1
        cur_attributes.append(attribute)
        cur_intervals.append([-np.inf, np.inf])

    if mode == 'left': # specify the intervals of new attribute
        cur_intervals[idx][1] = value
    elif mode == 'right':
        cur_intervals[idx][0] = value


def traverse(root, rules):
    # recursively traverse all the rules (leaf nodes) in binary decision tree
    # rules: A list that will store the rules (or information) from the leaf nodes of the tree
    # root: The current node being visited.
    # Initially, this will be the root node of the decision tree, but during the recursion, it will be any node in the tree.
    # rules.append(root.rule)
    if root.lchild is not None: traverse(root.lchild, rules)
    if root.rchild is not None: traverse(root.rchild, rules)
    if root.lchild is None and root.rchild is None:
        rules.append(root.rule)


# def cal_entropy_miss_outliers_gain(rule, columns, X, y):
#     """计算最优entropy gain/miss outliers gain"""
#
#     entropy_miss_outliers_gain = []
#     for col in columns:
#         for val in np.sort(np.unique(X[rule.idxes, col]))[:-1]:
#
#             new_avg_entropy = cal_split_entropy(X, y, rule, col, val)
#             new_miss_outliers = cal_split_miss_outliers(X, y, rule, col, val)
#
#             entropy_gain = new_avg_entropy - entropy(rule) * rule.num # entropy gain: delta entropy
#             miss_outliers_gain = new_miss_outliers - rule.miss_outliers
#
#             if entropy_gain <= 0: continue
#             if (miss_outliers_gain == 0) or (entropy_gain / miss_outliers_gain) < 0 : continue
#             entropy_miss_outliers_gain.append(entropy_gain / miss_outliers_gain)
#     # print(entropy_miss_outliers_gain)
#     if len(entropy_miss_outliers_gain) > 0:
#         entropy_miss_outliers_gain = np.array(entropy_miss_outliers_gain)
#         # entropy_miss_outliers_gain = entropy_miss_outliers_gain[entropy_miss_outliers_gain <= 0.1]
#         # print(entropy_miss_outliers_gain)
#         # if len(entropy_miss_outliers_gain) == 0:
#         #     return 0
#
#         idx = np.argmin(entropy_miss_outliers_gain)
#         return entropy_miss_outliers_gain[idx]
#     else:
#         return 0


def cal_length_miss_outliers_entropy_gain(rule, columns, X, y, max_length=15, penalty_term = None):
    """
    calculate the length to entropy ratio gain
    """
    length_miss_outliers_divide_entropy = []
    split_attributes = []
    split_nums = []

    assert rule.num == len(rule.idxes)

    entropys = []
    miss_outliers = []

    for col in columns:

        if col in rule.attributes:
            incremental_length = len(rule.attributes)
        else:
            incremental_length = len(rule.attributes) + 2 # increment by left and right nodes = 2
            if len(rule.attributes) + 1 > max_length: # if adding the new attribute to the rule exceed the length threshold
                continue

        for val in np.sort(np.unique(X[rule.idxes, col]))[:-1]: # search for smallest delta L / delta E

            new_avg_entropy = cal_split_entropy(X, y, rule, col, val)
            new_miss_outliers = cal_split_miss_outliers(X, y, rule, col, val)

            entropy_gain = new_avg_entropy - entropy(rule) * rule.num # entropy gain: delta entropy
            miss_outliers_gain = new_miss_outliers - rule.miss_outliers


            if entropy_gain <= 0  : continue

            entropys.append(new_avg_entropy)
            miss_outliers.append(new_miss_outliers)

            length_miss_outliers_divide_entropy.append((incremental_length + penalty_term * miss_outliers_gain) / entropy_gain)
            # (delta L + lambda * delta miss outliers) / delta E
            split_attributes.append(col)
            split_nums.append(val)

    if len(length_miss_outliers_divide_entropy) > 0:
        idx = np.argmin(np.array(length_miss_outliers_divide_entropy)) # return index of the smallest length to entropy ratio
        return length_miss_outliers_divide_entropy[idx], split_attributes[idx], split_nums[idx] # return best results
    else:
        return tuple([-1] * 3) # if length to entropy ratio is undefined = -1


def select(input_list, idxes):
    new_list = []
    for i, x in enumerate(input_list):
        if i in idxes:
            new_list.append(x)
    return new_list


def delete(input_list, idxes):
    new_list = []
    for i, x in enumerate(input_list):
        if i not in idxes:
            new_list.append(x)
    return new_list


def expand_rule(X, rule_to_expand, predictions, leaf_list, stats_list, stats, print_result=True):
    """
    rule expansion based on rule (leaf node) with smallest length to entropy ratio
    """
    if print_result:
        print(f"split rule:, split_num:{rule_to_expand.split_num}, split_attribute: {rule_to_expand.split_attribute}, miss_outliers:{rule_to_expand.miss_outliers}, predicted label: {rule_to_expand.label}, attributes: {rule_to_expand.attributes}")
    # split dataset into two subsets
    idxes_smaller = rule_to_expand.idxes[
        np.where(X[rule_to_expand.idxes, rule_to_expand.split_attribute] <= rule_to_expand.split_num)]
    idxes_larger = rule_to_expand.idxes[
        np.where(X[rule_to_expand.idxes, rule_to_expand.split_attribute] > rule_to_expand.split_num)]
    # majority vote for label of each subset
    label_1 = int(np.sum(predictions[idxes_smaller]) > len(idxes_smaller) / 2)
    label_2 = int(np.sum(predictions[idxes_larger]) > len(idxes_larger) / 2)

    predictions_1 = predictions[idxes_smaller]
    predictions_2 = predictions[idxes_larger]

    acc_1 = np.sum(predictions[idxes_smaller] == label_1) / len(idxes_smaller)
    acc_2 = np.sum(predictions[idxes_larger] == label_2) / len(idxes_larger)

    miss_outliers_1 = np.sum((predictions_1 == 1) & (np.full(len(predictions_1), label_1) == 0))
    miss_outliers_2 = np.sum((predictions_2 == 1) & (np.full(len(predictions_2), label_2) == 0))

    # define two decision nodes (rules) instances
    rule_1 = stats(len(idxes_smaller), acc_1, label_1, idxes_smaller,
                   list(dict.fromkeys(rule_to_expand.attributes + [rule_to_expand.split_attribute])), miss_outliers_1)
    rule_2 = stats(len(idxes_larger), acc_2, label_2, idxes_larger,
                   list(dict.fromkeys(rule_to_expand.attributes + [rule_to_expand.split_attribute])), miss_outliers_2)
    leaf_list.append(rule_1) # update leaf nodes list
    leaf_list.append(rule_2)
    stats_list.append(rule_1) # update rule list
    stats_list.append(rule_2)

    node_1 = Node()
    node_2 = Node()

    node_1.set_rule(rule_1) # create two child nodes for two new rules
    node_2.set_rule(rule_2)
    rule_1.set_node(node_1)
    rule_2.set_node(node_2)

    rule_to_expand.node.lchild = node_1 # left child node of the parent node (rule)
    rule_to_expand.node.rchild = node_2
    rule_to_expand.node.is_leaf = False


def expand_rule_and_add_to_list(leaf_list, stats_list, stats, rule, cur_C, columns, predictions,
                                max_length, avg_entropy0, total_length0, threshold, print_result=True):
    '''
    leaf_list: current leaf list, rule is not included in this list
    '''
    num_1 = rule.num
    expand_rule(X, rule, predictions, leaf_list, stats_list, stats, print_result=print_result)
    assert leaf_list[-1].num + leaf_list[-2].num == num_1 # check that the rule splitting is correct

    new_rules = leaf_list[-2:]
    new_idxes = [len(leaf_list) - 2, len(leaf_list) - 1]

    for rule, rule_idx in zip(new_rules, new_idxes): # for each new rule
        # compute the best length to entropy ratio and best splitting attribute and threshold
        rule.length_divide_entropy, rule.split_attribute, rule.split_num = \
            cal_length_entropy_gain(rule, columns, X, predictions, max_length=max_length)

        if not rule.length_divide_entropy == -1:
            rule_C = rule.length_divide_entropy * avg_entropy0 * total_length0 - sum(
                [len(r.attributes) for r in leaf_list]) # optimal M calculation

            if rule_C < cur_C:
                avg_entropy0, total_length0 = cal_list_entropy(leaf_list, return_detail=True)
                leaf_list = delete(leaf_list, [leaf_list.index(rule)]) # delete the current rule and expand rule into two new nodes
                expand_rule_and_add_to_list(leaf_list, stats_list, stats, rule, cur_C, columns, predictions,
                                            max_length, avg_entropy0, total_length0, threshold, print_result=True)

                f1_score_2 = cal_f1score(leaf_list)
                if f1_score_2 > threshold: return


def buildRuleTree(X, predictions, stats, cur_rule, stats_list, leaf_list, rows, columns, max_length, threshold,
                  penalty_term, print_result=True):
    '''
    X: Data Points
    predictions: no need to explain
    stas: namedtuple
    cur_rule: current_rule
    stats_list: rules except for the current rule
    rows: covered by cur_rule
    columns: covered by cur_rule
    '''

    C_along_the_process = [] # keep track of M value
    rule_num_along_the_process = []
    length_miss_outliers_entropy_gain_process = []
    objective_value = []

    cur_C = 0

    f1_score = cal_f1score(leaf_list)
    while f1_score < threshold:

        avg_entropy0, total_length0 = cal_list_entropy(leaf_list, return_detail=True) # used to compute A0

        cur_idx = -1

        leaf_C = []
        for idx, rule in enumerate(leaf_list):

            # if rule.entropy_miss_outliers_gain is None:  # compute l/e ratio, solitting feature and value for the expanded rule
            #     rule.entropy_miss_outliers_gain = cal_entropy_miss_outliers_gain(rule, columns, X, predictions)
            #     rule.penalty_term = (sum([r.miss_outliers for r in leaf_list]) * rule.entropy_miss_outliers_gain) / (
            #                     avg_entropy0 * total_length0)
            #     # if rule.penalty_term > 1:
            #     #     rule.penalty_term = np.clip(rule.penalty_term, 0, 1)

            # leaf_list is used to keep track of the current set of leaf nodes in the decision tree
            if rule.length_miss_outliers_divide_entropy is None: # compute l/e ratio, solitting feature and value for the expanded rule
                rule.length_miss_outliers_divide_entropy, rule.split_attribute, rule.split_num = \
                    cal_length_miss_outliers_entropy_gain(rule, columns, X, predictions, max_length=max_length, \
                                                            penalty_term = penalty_term)

            if rule.length_miss_outliers_divide_entropy == -1: # if length to entropy ratio is undefined
                leaf_C.append(np.inf)

            else:
                leaf_C.append(rule.length_miss_outliers_divide_entropy * avg_entropy0 * total_length0 - sum(
                        [len(r.attributes) for r in leaf_list]) - (penalty_term * sum([r.miss_outliers for r in leaf_list])))

        leaf_C = np.array(leaf_C)
        C_along_the_process.append(cur_C) # update the value of optimal M
        rule_num_along_the_process.append(len(leaf_list))

        if print_result:
            print("C = ", cur_C, "f1 score:", f1_score)

        # objective_value.append(avg_entropy0 * total_length0 / (sum(
        #     [len(r.attributes) for r in leaf_list]) + (
        #                                    penalty_term * sum([r.miss_outliers for r in leaf_list])) + cur_C))

        idxes = np.where(leaf_C <= cur_C)[0]

        if len(idxes) > 0: # only used for root rule, special case

            rules_to_expand = select(leaf_list, idxes) # select the rules that need to be expanded
            leaf_list = delete(leaf_list, idxes) # delete those expanded rules from leaf list
            # if we want to keep track with the rules, then we cannot delete the splitted rules

            flag = 0
            for r_i, r in enumerate(rules_to_expand):
                expand_rule(X, r, predictions, leaf_list, stats_list, stats, print_result=print_result)
                f1_score = cal_f1score(leaf_list + rules_to_expand[r_i:])
                if f1_score >= threshold:
                    flag = 1
                    break # break the inner loop
            if flag: break # break the outer loop

        elif (leaf_C == np.inf).all(): # when no further rule expansion can be made
            break

        else: # this means that M0 is greater than Mb, the boundary M value, which can produce valid split
            idx = np.argmin(leaf_C) #select the rule with smallest M value
            cur_C = leaf_C[idx]

            # remove idx in leaf_list
            rule_to_expand = leaf_list[idx]
            del leaf_list[idx]
            # split the rule with M slightly larger than M0
            expand_rule(X, rule_to_expand, predictions, leaf_list, stats_list, stats, print_result=print_result)

        f1_score = cal_f1score(leaf_list)

    if print_result:
        print("Final F-1 score:", cal_f1score(leaf_list))
        print("Final recall score", cal_recall(leaf_list))
        print("Final precision score", cal_precision(leaf_list))

    C_along_the_process.append(cur_C)
    rule_num_along_the_process.append(len(leaf_list))
    if print_result:
        print(f'The optimal M value along the process : {C_along_the_process}')
        print(f"rule number along the process : {rule_num_along_the_process}")
        # print(f"Objective value along the process: {objective_value}")

    return leaf_list, C_along_the_process, rule_num_along_the_process


def draw_C(rule_num, C, name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")
    import pandas as pd
    df = pd.DataFrame({"Rule Number": rule_num, "C": C, "color": 'blue'})
    sns.set_theme(style="darkgrid")
    ax = sns.lineplot(x='Rule Number', y='C', data=df, markers=True, marker='o', color='blue')
    plt.savefig(f"../increase_of_C_{name}.png", dpi=600)


def calculate(X, predictions, criterion=1, max_length=10, max_depth=-1, threshold=0.90, penalty_term = 0.1, print_result=True,
              return_rule_num=False):
    print("Threshold:", threshold)
    class stats:
        def __init__(self, num, acc, label, idxes, attributes, miss_outliers):
            self.num = num
            self.acc = acc
            self.label = label
            self.idxes = idxes
            self.attributes = attributes
            self.miss_outliers = miss_outliers # add new attribute
            self.length_miss_outliers_divide_entropy = None
            # self.entropy_miss_outliers_gain = None
            # self.penalty_term = None
            self.split_attribute = None
            self.split_num = None

        def set_node(self, node): # a node is linked to a rule, a rule is linked to a node
            self.node = node

        def extra_repr(self):
            return f"num = {self.num}, acc = {self.acc}, label = {self.label}, attributes = {self.attributes}, miss_outliers = {self.miss_outliers}"

    stats_list = []
    rows = np.arange(X.shape[0])
    columns = np.arange(X.shape[1])
    # since we let r0 at the root node to be that all data points belong to inliners, so missed outliers are the
    # portion of dataset that has label = 1
    root_rule = stats(len(X), np.sum(predictions == 0) / len(predictions), np.sum(predictions) // len(predictions),
                      rows, [], np.sum(predictions))
    root = Node()
    root_rule.set_node(root)
    root.set_rule(root_rule)
    stats_list.append(root_rule)
    leaf_list = [root_rule]

    leaf_list, C_list, rule_num_along_process = buildRuleTree(X, predictions, stats, root_rule, stats_list, leaf_list, rows, columns, max_length,
                              threshold, penalty_term, print_result=print_result)
    if root.is_leaf: root.rule = root_rule

    rules = leaf_list

    if print_result:
        print("rules F-1 score", cal_f1score(rules))
        # print("rules F-1 score", cal_f1score(leaf_list))
        print("whole lengths:", np.sum([len(r.attributes) for r in rules]))
        print('lengths:', [len(r.attributes) for r in rules])
        print("Total Rule number:", len(rules))

    if return_rule_num:
        return root, np.sum([len(r.attributes) for r in rules]), len(rules)
    else:
        return root # return the root node of the rule based tree


def main(name, lof_predictions, lof_scores, ratio, max_length, threshold, penalty_term, method='MDT'):

    # X, y, lof_krange, N_range, knn_krange, if_range, mahalanobis_N_range = load_data(name, data_dir=r'D:\ANow\data/')

    y = np.zeros(len(X))

    if method == 'MDT':
        # Random subsample
        if use_lof:
            outlier_idxes = np.where(lof_predictions == 1)[0]
            Outliers = X[outlier_idxes]
            # label_outliers = label_true[outlier_idxes] # select the correponding ground truth labels from label_true arg

            idxes = np.where(lof_predictions == 0)[0]
            selected_indices = np.random.choice(np.arange(len(idxes)), size=int(len(idxes) * ratio), replace=False)
            Inliers = X[idxes[selected_indices]]
            # label_inliers = label_true[idxes[selected_indices]] # select the correponding ground truth labels from label_true arg

            X_new = np.concatenate([Outliers, Inliers], axis=0)
            lof_predictions_new = np.concatenate([np.ones(len(Outliers)), np.zeros(len(Inliers))], axis=0)
            # label_true_new = np.concatenate([label_inliers, label_inliers], axis=0)
            # transformer = RobustScaler().fit(X_new)
            # X_new_transformed = transformer.transform(X_new)

            import time
            start_time = time.time()
            # run build rule based decision tree
            root, length, num = calculate(X_new, lof_predictions_new, max_length=max_length, criterion=1, print_result=True, threshold=threshold, penalty_term=penalty_term, return_rule_num=True)
            end_time = time.time()

            predictions = root.predict(X_new)
            f1_score = metrics.f1_score(lof_predictions_new, predictions)
            recall_score = metrics.recall_score(lof_predictions_new, predictions)
            time = end_time - start_time
            # print("Final F1 score:", f1_score)
            # print("Final Recall score:", recall_score)
            print("Time consumed:", time)
            return f1_score, time, length, num

        elif use_misdetect:
            import time
            start_time = time.time()
            # run build rule based decision tree
            root, length, num = calculate(X, misdetect_predictions, max_length=max_length, criterion=1,
                                          print_result=True, threshold=threshold, return_rule_num=True, penalty_term=penalty_term)
            end_time = time.time()

            predictions = root.predict(X)
            f1_score = metrics.f1_score(lof_predictions, predictions)
            time = end_time - start_time
            print("Final F1 score:", f1_score)
            rules = []
            traverse(root, rules)
            print([(r.attributes, r.split_attribute, r.split_num, r.label) for r in rules])
            print("Time consumed:", time)
            return f1_score, time, length, num

        else: # use original data to build STAIR rule tree
            import time
            start_time = time.time()
            # run build rule based decision tree
            root, length, num, rules = calculate(X, raw_label, max_length=max_length, criterion=1,
                                          print_result=True, threshold=threshold, return_rule_num=True)
            end_time = time.time()

            predictions = root.predict(X)
            f1_score = metrics.f1_score(lof_predictions, predictions)
            time = end_time - start_time
            print("Final F1 score:", f1_score)
            print("Time consumed:", time)
            return f1_score, time, length, num


    elif method == 'CART':
        clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
        path = clf.cost_complexity_pruning_path(X, lof_predictions)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        for ccp_alpha in ccp_alphas[::-1]:
            clf = DecisionTreeClassifier(criterion='entropy', random_state=0, ccp_alpha=ccp_alpha)
            clf.fit(X, lof_predictions)
            if metrics.f1_score(lof_predictions, clf.predict(X)) > threshold:
                break
        leaf_nodes = clf.apply(X)
        lengths = []
        from sklearn.tree import _tree
        def tree_to_code(tree, feature_names):
            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]
            print("def tree({}):".format(", ".join(feature_names)))

            def recurse(node, depth, attributes):
                indent = "  " * depth
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    # print("{}if {} <= {}:".format(indent, name, threshold))
                    attributes = set(attributes + [name])

                    recurse(tree_.children_left[node], depth + 1, list(attributes))
                    # print("{}else:  # if {} > {}".format(indent, name, threshold))

                    recurse(tree_.children_right[node], depth + 1, list(attributes))
                else:
                    # print("{}return {}".format(indent, tree_.value[node]))
                    lengths.append(len(attributes))

            recurse(0, 1, [])

        tree_to_code(clf, [str(i) for i in np.arange(X.shape[1])])
        # print(lengths)
        f1_score = metrics.f1_score(lof_predictions, clf.predict(X))
        print("final F-1 score:", f1_score)
        print("whole lengths:", np.sum(lengths))
        print("lengths:", lengths)
        print("Total rule Number:", len(lengths))
        return f1_score, 0, np.sum(lengths), len(lengths)

    else:
        for depth in range(7, 25):
            clf = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=depth)
            # clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
            clf.fit(X, lof_predictions)
            leaf_nodes = clf.apply(X)
            lengths = []
            from sklearn.tree import _tree
            def tree_to_code(tree, feature_names):
                tree_ = tree.tree_
                feature_name = [
                    feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                    for i in tree_.feature
                ]
                # print("def tree({}):".format(", ".join(feature_names)))

                def recurse(node, depth, attributes):
                    indent = "  " * depth
                    if tree_.feature[node] != _tree.TREE_UNDEFINED:
                        name = feature_name[node]
                        threshold = tree_.threshold[node]
                        # print("{}if {} <= {}:".format(indent, name, threshold))
                        attributes = set(attributes + [name])

                        recurse(tree_.children_left[node], depth + 1, list(attributes))
                        # print("{}else:  # if {} > {}".format(indent, name, threshold))

                        recurse(tree_.children_right[node], depth + 1, list(attributes))
                    else:
                        # print("{}return {}".format(indent, tree_.value[node]))
                        lengths.append(len(attributes))

                recurse(0, 1, [])

            tree_to_code(clf, [str(i) for i in np.arange(X.shape[1])])
            # print(lengths)
            f1_score = metrics.f1_score(lof_predictions, clf.predict(X))
            print(f"Depth: {depth}, f1_score: {f1_score}")
            if f1_score > threshold: break
        print("final F-1 score:", f1_score)
        print("whole lengths:", np.sum(lengths))
        print("lengths:", lengths)
        print("Total rule Number:", len(lengths))
        return f1_score, 0, np.sum(lengths), len(lengths)

if __name__ == '__main__':
    # sys.argv is a list that contains the command-line arguments passed to the script
    if len(sys.argv) > 1:
        if sys.argv[1] == 'All':
            names = ['Pendigits', 'Pima', "Mammography", "Satimage-2", "PageBlock", "satellite", "ALOI", "optdigits", "Annthyroid", "skin"]
        else:
            names = [sys.argv[1]]
    else: # if no argument provided, use default pendigits dataset
        names = ["Pendigits"]

    # 根据实际路径修改
    data_dir = r'../data/'
    # max_lengths = [2, 4, 6, 8, 10, 12]
    max_lengths = [10]
    # methods = ['DT', 'CART', 'MDT']
    methods = ['MDT']
    thresholds = [0.8]
    # thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    penalty_terms = [0.01]
    results = []
    use_lof = 1 # specify whether use lof or rrl to make prediction
    use_misdetect = 0 # specify whether to use misdetect module
    # df_ml = pd.DataFrame(columns = names, index = max_lengths)
    # df_f1 = pd.DataFrame(columns = names, index = thresholds)


    for name in names:

        print(name)

        if use_lof:

            data_dir = r'../data/'

            X, y, lof_krange, N_range, knn_krange, if_range, mahalanobis_N_range = load_data(name,
                                                                                             data_dir=data_dir)
            X, y = np.array(X), np.array(y)

            def run_lof(X, y, num_outliers=560, k=60):
                clf = LocalOutlierFactor(n_neighbors=k)
                clf.fit(X)
                lof_scores = -clf.negative_outlier_factor_
                threshold = np.sort(lof_scores)[::-1][num_outliers]
                lof_predictions = np.array(lof_scores > threshold)
                lof_predictions = np.array([int(i) for i in lof_predictions])
                print("F-1 score of LOF outlier detection:", metrics.f1_score(y, lof_predictions))
                return lof_predictions, lof_scores

            lof_predictions, lof_scores = run_lof(X, y, k=lof_krange[7], num_outliers=int(np.sum(y)))
            df_m = pd.DataFrame(columns=['M_value', 'Rule_num'])

        elif use_misdetect:
            print('running on cleaned dataset')
            data_dir = r'../cleaned_data/'
            X, y = load_misdetect_prediction(name=name, data_dir=data_dir)
            X, y = np.array(X), np.array(y)
            misdetect_predictions = y

        else: # test case where i use original data to fit the rule tree STAIR
            data_dir = r'../data/'
            X, y, lof_krange, N_range, knn_krange, if_range, mahalanobis_N_range = load_data(name,
                                                                                             data_dir=data_dir)
            X, y = np.array(X), np.array(y)
            raw_label = y

        for method in methods:
            for ratio in [1.0]:
                for max_length in max_lengths:
                    for threshold in thresholds:
                        for penalty_term in penalty_terms:
                            print('Penalty term', penalty_term)
                            f1_scores = []
                            times = []
                            lengths = []
                            nums = []
                            for i in range(1):
                                if use_lof: # use lof predictions to do STAIR
                                    print('using lof predictions')
                                    result = main(name, lof_predictions, lof_scores, ratio, max_length, threshold, penalty_term, method)
                                    # print("max length:" , max_length)
                                    f1_scores.append(result[0])
                                    times.append(result[1])
                                    lengths.append(result[2])
                                    nums.append(result[3])
                                    results.append((name, ratio, max_length, np.mean(f1_scores), np.mean(times), np.mean(lengths), np.mean(nums)))
                                    # print(result[4])
                                    # df_m.loc[:, 'M_value'] = result[4]
                                    # df_m.loc[:,'Rule_num'] = result[5]
                                    # print('---saving M values---')
                                    # df_m.to_csv(f'./experiment_data/M_stair_{name}.csv')
                                    # print(df_m)
                                    print(results)


                                elif use_misdetect: # use rrl predictions to do STAIR
                                    lof_scores = None
                                    print('total outlier num', np.sum(misdetect_predictions))
                                    result = main(name, misdetect_predictions, lof_scores, ratio, max_length, threshold, penalty_term, method)
                                    # print("max length:" , max_length)
                                    f1_scores.append(result[0])
                                    times.append(result[1])
                                    lengths.append(result[2])
                                    nums.append(result[3])
                                    results.append((name, ratio, max_length, np.mean(f1_scores), np.mean(times), np.mean(lengths),
                                                    np.mean(nums)))
                                    print(results)

                                else:
                                    lof_scores = None
                                    result = main(name, raw_label, lof_scores, ratio, max_length, threshold, method)
                                    # print("max length:" , max_length)
                                    f1_scores.append(result[0])
                                    times.append(result[1])
                                    lengths.append(result[2])
                                    nums.append(result[3])
                                    results.append(
                                        (name, ratio, max_length, np.mean(f1_scores), np.mean(times), np.mean(lengths),
                                         np.mean(nums)))
                                    print(results)







