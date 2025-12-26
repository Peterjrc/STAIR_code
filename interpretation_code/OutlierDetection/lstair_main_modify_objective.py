import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from load_data import load_data
from main_modify_objective import calculate, traverse
from config import load_default_gap
import math
from utils import load_misdetect_prediction
# from local_search_outlier import *

sns.set_theme(style='darkgrid')


class Equal_Size_Kmeans():
    """
    create equal size kmeans clusters with recursion splitting
    fit() train equal size kmeans
    predict( ) recursively predict using trained equal size kmeans

    """
    def __init__(self, repeat_times=3):
        self.repeat_times = repeat_times
        self.clusters = []
        self.kmeans = KMeans(n_clusters=2) # kmeans
        if repeat_times > 1:
            for i in range(2): # recursively doing equal size kmeans
                self.clusters.append(Equal_Size_Kmeans(repeat_times=repeat_times - 1))

    def fit(self, X):
        self.labels_ = np.zeros(len(X))
        # Fit data with KMeans
        self.kmeans.fit(X)

        # Adjust the labels to balance them
        self.labels_ = self.kmeans.labels_
        unique_labels, counts = np.unique(self.labels_, return_counts=True) # count the unique labels for 2 clusters
        if counts[0] < counts[1]:
            self.small_group = 0
            # pass args in opposite position
            idxes, self.delta = self.label_to_flip(self.kmeans.cluster_centers_[unique_labels[0]],
                                                   self.kmeans.cluster_centers_[unique_labels[1]],
                                                   X[np.where(self.labels_ == unique_labels[1])],
                                                   counts[1] - len(X) // 2)
            idxes = np.where(self.labels_ == unique_labels[1])[0][idxes] # select indces in cluster 1 need to be flipped
            self.labels_[idxes] = unique_labels[0] # flip lables from cluser 1 to 0
        elif counts[0] > counts[1]:
            self.small_group = 1
            idxes, self.delta = self.label_to_flip(self.kmeans.cluster_centers_[unique_labels[1]],
                                                   self.kmeans.cluster_centers_[unique_labels[0]],
                                                   X[np.where(self.labels_ == unique_labels[0])],
                                                   counts[0] - len(X) // 2)
            self.labels_[np.where(self.labels_ == unique_labels[0])[0][idxes]] = unique_labels[1]

        # Continue splitting
        if self.repeat_times > 1:
            for i in [1, 0]: # for original clusters 1 and 2
                idxes = np.where(self.labels_ == i)
                # create 2 new clusters
                self.clusters[i].fit(X[idxes])
                # modify the labels to avoid repetition
                self.labels_[idxes] = self.clusters[i].labels_ + i * 2 ** (self.repeat_times - 1) # update labels for new clusters

        return self

    def predict(self, X):
        # intial clusters produced by trained kmeans

        labels = self.kmeans.predict(X)

        if self.repeat_times > 1:
            distances0 = np.sum((X - self.kmeans.cluster_centers_[0].reshape(1, -1)) ** 2, axis=1)
            distances1 = np.sum((X - self.kmeans.cluster_centers_[1].reshape(1, -1)) ** 2, axis=1)
            deltas = distances0 - distances1

            if self.small_group == 0:
                # if small_group is 0, then we should let idxes0 become larger
                idxes0 = np.where(deltas <= self.delta)[0]
                idxes1 = np.where(deltas > self.delta)[0]
            else:
                idxes0 = np.where(deltas <= - self.delta)[0]
                idxes1 = np.where(deltas > - self.delta)[0]

            for i in [1, 0]:
                idxes = eval(f"idxes{i}")
                # shift the labels for newly created clsuters when i = 1
                # recursively predict
                labels[idxes] = self.clusters[i].predict(X[idxes]) + i * 2 ** (self.repeat_times - 1)

        # if self.repeat_times > 1:
        #     for i in [1, 0]:
        #         idxes = np.where(labels == i)
        #         labels[idxes] = self.clusters[i].predict(X[idxes]) + i * 2 ** (self.repeat_times-1)
        return labels

    def label_to_flip(self, center0, center1, X, num):
        '''
        params:
        center0: the center of the cluster that has more points
        center1: the center of the cluster that has fewer points
        X: the data in cluster 0
        num: the number of points that should be transferred from cluster0 to cluster1

        return:
        the idxes of X that should be flipped
        '''
        distances0 = np.sum((X - center0.reshape(1, -1)) ** 2, axis=1) # distance from all x to center0
        distances1 = np.sum((X - center1.reshape(1, -1)) ** 2, axis=1)
        deltas = distances0 - distances1 # deltas < 0: closer to center 0, >0: closer to center 1
        idxes = np.argsort(deltas)[:num] # argsort will return the indces of sorted array, select top smallest num of them to flip
        delta = deltas[idxes[-1]] # boundary delta
        return idxes, delta


def draw_cluster(cluster, color):
    sns.scatterplot(cluster[:, 0], cluster[:, 1], c=[color] * len(cluster))


def draw_classifier(classifier, color):
    a, b = classifier.coef_
    c = classifier.intercept_

    if b != 0:
        # a*x + b*y + c = 0 ==> y = (-ax-c)/b
        x = np.linspace(-8, 8, 2000)
        y = - (a * x + c) / b
    else:
        if a == 0:
            print("a = 0, b = 0")
            return
        else:
            x = np.array([-c / a] * 150)
            y = np.linspace(-8, 8, 150)

    if np.max(abs(y)) > 8:
        idxes = np.where((y < 8) & (y > -8))
        y = y[idxes]
        x = x[idxes]

    plt.plot(x, y, color=color)


class LocalDT():
    def __init__(self, num, device='cpu'):
        self.num = num
        self.decision_trees = None
        self.cluster = None
        self.centers = None
        self.predictions = None
        self.device = device

    def update_decision_tree(self, X, y, threshold, penalty_term, penalty_cluster_idx = None):
        """
        return a list of rule decision tree, where each element in the list is the root node of each tree for every cluster
        """
        #

        if penalty_cluster_idx is not None: # whether use penaly term for infeiror cluster
            self.decision_trees = []
            # penalty_cluster = self.clusters[penalty_cluster_idx]
            # other_clusters = [idxes for i, idxes in enumerate(self.clusters) if i != penalty_cluster_idx]

            for i, idxes in enumerate(self.clusters): # for instances in each cluster, update the decision tree

                # clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
                # path = clf.cost_complexity_pruning_path(X[idxes], y[idxes])
                # ccp_alphas, impurities = path.ccp_alphas, path.impurities
                #
                # for ccp_alpha in ccp_alphas[::-1]:
                #     clf = DecisionTreeClassifier(criterion='entropy', random_state=0, ccp_alpha=ccp_alpha)
                #     clf.fit(X[idxes], y[idxes])
                #     if metrics.f1_score(y[idxes], clf.predict(X[idxes])) > threshold:
                #         break
                # self.decision_trees.append(clf)
                # fit the rule tree for each cluster
                if i == penalty_cluster_idx:
                    print(f'penalizing cluster {i}')
                    self.decision_trees.append(
                        calculate(X[idxes], y[idxes], max_length=max_length_global, threshold=threshold, penalty_term=penalty_term,return_rule_num=False,print_result=False))
                else:
                    self.decision_trees.append(
                        calculate(X[idxes], y[idxes], max_length=max_length_global, threshold=threshold, penalty_term=0,
                                  return_rule_num=False, print_result=False))
        else:
            self.decision_trees = []
            for idxes in self.clusters:
                self.decision_trees.append(
                    calculate(X[idxes], y[idxes], max_length=max_length_global, threshold=threshold, print_result=False, penalty_term=0, return_rule_num=False))


    def update_clusters(self, X, y):
        '''
        params: X: features, y: labels
        return: new clusters
        '''

        idx2clusters = np.zeros(len(X))
        for i in range(len(self.clusters)):
            idx2clusters[self.clusters[i]] = i

        new_clusters = []
        for i in range(len(self.clusters)):
            new_clusters.append([]) # initialize new clusters for each original cluster

        all_preds = []
        for i, dt in enumerate(self.decision_trees):
            assert len(self.decision_trees) == len(self.clusters)# make predictions using all rule trees for all X
            all_preds.append(dt.predict(X)), 'number does not match!'

        for idx in range(len(X)): # for every instance in X
            classify_correct = np.zeros(len(self.clusters)) # initialize correctly classification num for every cluster/tree

            for i, dt in enumerate(self.decision_trees): # examine the prediction accuracy for each instance in X
                # pdb.set_trace()
                if all_preds[i][idx] == y[idx]: # for the i th rule tree and idx th instance
                    # if dt.predict_single(X[idx]) == y[idx]:
                    classify_correct[i] += 1

            distances = np.zeros(len(self.clusters)) # initalize euclidean distance for each cluster
            for i, c in enumerate(self.centers):
                distances[i] = np.sqrt(np.sum((X[idx] - c) ** 2) / X.shape[1]) # locality calculation

            cluster_score = 5 * distances - classify_correct # objective function for each cluster for single instance xi

            new_clusters[np.argmin(cluster_score)].append(idx) # select the cluster with smallest objective function value,
            # append the instance x to that cluster
            # update the clusters according to ojective equation (24)
            # if np.argmin(cluster_score) != idx2clusters[idx]:
            #     print("wrong")

    # in the end, each list in new clusters will correspond to one cluster which contains the assgined instances in X
        self.clusters = [] # redefine the clusters
        new_decision_trees = []
        for cluster, dt in zip(new_clusters, self.decision_trees):
            if len(cluster) > 0:
                new_decision_trees.append(dt)
                self.clusters.append(cluster) # update the new clusters
        # self.clusters = new_clusters
        # re-calculate the centroids for each cluster
        self.centers = [np.mean(X[idxes], axis=0) for idxes in self.clusters]
        self.decision_trees = new_decision_trees # redefine the rule tree


    def calculate_loss_acc(self, X, y):
        # calculate f1 score after updating the clusters, need to compare with threshold f1
        predictions = []
        y_true = []
        f1_scores = []
        recall_scores = []
        for i, idxes in enumerate(self.clusters):  # calculate f1 score for each cluster
            predictions.append(self.decision_trees[i].predict(X[idxes]))  # rule decision tree predictions for each cluster
            y_true.append(y[idxes])  # ground truth label
            if len(np.unique(predictions[-1])) == 1 and len(
                    np.unique(y[idxes])) == 1:  # special case when all predicted labels are the same
                f1_scores.append(1 if np.unique(predictions[-1])[0] == np.unique(y[idxes])[0] else 0)
                recall_scores.append(1 if np.unique(predictions[-1])[0] == np.unique(y[idxes])[0] else 0)
            else:
                f1_scores.append(
                    metrics.f1_score(y[idxes], predictions[-1]))  # calculate f1 score for each cluster prediction
                recall_scores.append(metrics.recall_score(y[idxes], predictions[-1]))

        y_true = np.concatenate(y_true)
        predictions = np.concatenate(predictions)
        return np.array(f1_scores), metrics.f1_score(y_true, predictions), np.array(recall_scores), metrics.recall_score(y_true, predictions)


    def initialize_clusters(self, X, n_clusters, random_state=0):
        """
        return a list of clusters, where each cluster (list) contain some instances in X
        """
        # assert n_clusters == 2
        # For the other datasets
        # kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)

        # For Thursday-01-03 and Shuttle and ALOI:
        kmeans = Equal_Size_Kmeans(repeat_times=int(math.log(n_clusters, 2))).fit(X)

        clusters = []
        clusters_idxes = []
        for _ in range(n_clusters):
            clusters.append([])
            clusters_idxes.append([])
        for idx, (x, label) in enumerate((zip(X, kmeans.labels_))):
            clusters[label].append(idx) # mapping each instance index to its corresponding cluster
        return clusters, kmeans


    def delete(self, clusters, idxes):
        new_clusters = []
        for i in range(len(clusters)):
            if i not in idxes:
                new_clusters.append(clusters[i])
        return new_clusters

    def calculate_cluster_entropy(self,X,y, clusters):
        class_ratio_normal = np.array([np.sum(y[idxes] == 0)/len(y[idxes]) for i, idxes in enumerate(clusters)])
        class_ratio_outlier = np.array([np.sum(y[idxes] == 1)/len(y[idxes]) for i, idxes in enumerate(clusters)])
        clusters_entropy = -class_ratio_normal * np.log(class_ratio_normal + 1e-8) - class_ratio_outlier * np.log(class_ratio_outlier+1e-8)
        return clusters_entropy

    def run(self, X, y, threshold, penalty_term, gap=0.2):
        # initialize the kmeans clusters

        # clusters = [np.concatenate(c) for c in clusters]
        clusters, _ = self.initialize_clusters(X, self.num) # intialize clusters
        self.clusters = clusters
        self.centers = [np.mean(X[idxes], axis=0) for idxes in self.clusters] # compute centroids for each cluster

        print("number of instances in each cluster:", [len(c) for c in self.clusters])

        # run the process via iteration
        # for i in range(10000):
        count = 0
        while True:
            count += 1
            i = len(self.clusters) # number of clusters
            # for all the other datasets:
            if count > 1:
                print('penalizing cluster')
                self.update_decision_tree(X, y, threshold=(threshold - 0.2) + 0.2 / 9 * (i-2), penalty_term= penalty_term, penalty_cluster_idx = penalty_cluster_idx)
            else:
                self.update_decision_tree(X, y, threshold=(threshold - 0.2) + 0.2 / 9 * (i-2), penalty_term= 0)

            # for Thursday-01-03:
            # if count > 1 and (penalty_cluster_idx != idx):
            #     self.update_decision_tree(X, y, threshold=(threshold - 0.1) + 0.1/ 9 * (i-2), penalty_term= penalty_term, penalty_cluster_idx = penalty_cluster_idx)
            # else:
            #     self.update_decision_tree(X, y, threshold=(threshold - 0.1) + 0.1 / 9 * (i-2), penalty_term= 0)
            # self.update_decision_tree(X, y, threshold=(threshold - 0.1) + 0.1 / 9 * (i-2), penalty_term= penalty_term)

            # for cover with n = 4:
            # self.update_decision_tree(X, y, threshold=(threshold - 0.6) + 0.6 / 9 * (i-2), penalty_term= penalty_term)

            # for cover with n = 8:
            # self.update_decision_tree(X, y, threshold=0.4, penalty_term= penalty_term)

            # for PageBlock:
            # fit the rule decision tree for each cluster
            # if count > 1:
            #     self.update_decision_tree(X, y, threshold=(threshold - 0.3) + 0.3 / 9 * (i-2), penalty_term= penalty_term, penalty_cluster_idx = penalty_cluster_idx)
            # else:
            #     self.update_decision_tree(X, y, threshold=(threshold - 0.3) + 0.3 / 9 * (i-2), penalty_term= 0)

            # self.update_decision_tree(X, y, threshold=(threshold - 0.3) + 0.3 / 9 * (i - 2), penalty_term= penalty_term)

            # for shuttle:
            # if count > 1 and (penalty_cluster_idx != idx):
            #     self.update_decision_tree(X, y, threshold=(threshold - 0.4) + 0.4/ 9 * (i-2), penalty_term= penalty_term, penalty_cluster_idx = penalty_cluster_idx)
            # else:
            #     self.update_decision_tree(X, y, threshold=(threshold - 0.4) + 0.4 / 9 * (i-2), penalty_term= 0)
            # self.update_decision_tree(X, y, threshold=(threshold - 0.4) + 0.4 / 9 * (i - 2), penalty_term=penalty_term)

            # for Friday-02-03:
            # self.update_decision_tree(X, y, threshold=(threshold - 0.0) + 0.0 / 9 * (i-2), penalty_term= penalty_term)

            self.update_clusters(X, y) # update clusters based on objective equation (24), return the new clusters

            f1_scores, f1_score, recall_scores, recall_score = self.calculate_loss_acc(X,y)  # calculate f1 score and overall f1 score for clusters
            sys.stdout.flush()
            print(f"epoch: {i}, F1 score: {f1_score}")
            # penalty_cluster_idx = np.argmin(recall_scores)

            if f1_score > threshold or i > 10: # break condition
                # f1 score larger than threshold or ietration (number of clusters) > 10
                print("f1 scores for every cluster:", f1_scores)
                print("recalls for every cluster:", recall_scores)
                print("overall f1 score:", f1_score)
                print("overall recall score:", recall_score)
                break

            clusters_to_delete = []

            if (f1_scores < 0.8).any():
                idx = np.argmin(f1_scores)
                print("cluster with smallest f1 score:", idx)
                # if penalty_cluster_idx == idx:
                #     print('penalizing and splitting the same cluster')
                # for idx in np.where(f1_scores < 0.95)[0]:
                if len(self.clusters[idx]) <= self.num:
                    continue
                clusters, kmeans = self.initialize_clusters(X[self.clusters[idx]], 2)
                # break the orginal cluster with insufficient f1 score into 2 new clusters
                clusters = [np.array(self.clusters[idx])[tmp_idxes].tolist() for tmp_idxes in clusters]

                centers = [np.mean(X[idxes], axis=0) for idxes in clusters]
                # for cluster_i,  cluster in enumerate(clusters):
                #     for x_i in cluster:
                #         if not np.sum((X[x_i] - centers[cluster_i]) ** 2) < np.sum((X[x_i] - centers[1 - cluster_i])**2):
                #             print("wrong")

                self.clusters.extend(clusters)
                clusters_to_delete.append(idx)
                self.centers.extend(centers)
                # break

            self.clusters = self.delete(self.clusters, clusters_to_delete)
            self.centers = self.delete(self.centers, clusters_to_delete)
            # self.decision_trees = self.delete(self.decision_trees, clusters_to_delete)
            outlier_num_count = [np.sum(y[idxes] == 1) for i, idxes in enumerate(self.clusters)]
            outlier_percentage = [np.sum(y[idxes] == 1) / len(idxes) for i, idxes in enumerate(self.clusters)]
            entropy_cluster = self.calculate_cluster_entropy(X,y,self.clusters)

            penalty_cluster_idx = np.argmax(outlier_num_count) # for others
            # penalty_cluster_idx = np.argmax(entropy_cluster) # pima, mammography(sum)
            # penalty_cluster_idx = np.argmin(outlier_percentage) # for PageBlock, Annthyroid
            # penalty_cluster_idx = np.argmax(cluster_num_count)

            print("penalized cluster has instances of", len(self.clusters[penalty_cluster_idx]))
            # print("penalized cluster has percentage of outliers", cluster_num_count[penalty_cluster_idx])
            print("cluster need to penalize in the next iter:", penalty_cluster_idx)
        # np.save(f"./{name}.npy", np.array(self.clusters))
        print("==" * 10)

        num = 0
        length = 0

        # all_lengths = []
        all_rule_nums = []
        all_rule_lengths = []
        all_cluster_lengths = []

        for i in range(len(self.clusters)):
            rules = []
            traverse(self.decision_trees[i], rules) # extract all the rule from each rule tree for each cluster
            all_rule_lengths.append(np.sum([len(r.attributes) for r in rules]))
            all_rule_nums.append(len(rules))
            num += len(rules)
            length += np.sum([len(r.attributes) for r in rules])
            # print("whole rule lengths:", np.sum([len(r.attributes) for r in rules]))
            # print('lengths:', [len(r.attributes) for r in rules])
            # print("Total Rule number:", len(rules))
            # print("cluster length:", len(self.clusters[i]))

            # lengths = []
            # from sklearn.tree import _tree
            # def tree_to_code(tree, feature_names):
            #     tree_ = tree.tree_
            #     feature_name = [
            #         feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            #         for i in tree_.feature
            #     ]
            #     # print("def tree({}):".format(", ".join(feature_names)))
            #
            #     def recurse(node, depth, attributes):
            #         indent = "  " * depth
            #         if tree_.feature[node] != _tree.TREE_UNDEFINED:
            #             name = feature_name[node]
            #             threshold = tree_.threshold[node]
            #             # print("{}if {} <= {}:".format(indent, name, threshold))
            #             attributes = set(attributes + [name])
            #
            #             recurse(tree_.children_left[node], depth + 1, list(attributes))
            #             # print("{}else:  # if {} > {}".format(indent, name, threshold))
            #
            #             recurse(tree_.children_right[node], depth + 1, list(attributes))
            #         else:
            #             # print("{}return {}".format(indent, tree_.value[node]))
            #             lengths.append(len(attributes))
            #
            #     recurse(0, 1, [])
            #
            # tree_to_code(self.decision_trees[i], [str(i) for i in np.arange(X.shape[1])])
            # num += len(lengths)
            # length += np.sum(lengths)
            # all_rule_lengths.append(np.sum(lengths))
            # all_rule_nums.append(len(lengths))

            all_cluster_lengths.append(len(self.clusters[i]))

        print(pd.DataFrame({
            "rule_number": all_rule_nums,
            "rule_length": all_rule_lengths,
            "cluster_number": all_cluster_lengths
        }))

        print(f"In all, the number is: {num}, length is: {length}")
        return num, length, len(self.clusters)

        # print("Final cluster number:", len(self.boundaries.clusters))
        # for i, cluster in enumerate(self.boundaries.clusters):
        #     print(f"cluster {i}, length: {len(cluster)}")

    def draw(self, X):
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)

        fig = plt.figure(figsize=(7, 6))
        fig.tight_layout()
        # xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
        #                      np.linspace(-7, 7, 150))

        # predictions = (self.predictions > 0).astype(int)
        colors = np.array(['#377eb8', '#ff7f00', '#0000FF', '#FFFF00',
                           '#FFA500', '#FF0000', '#008000', '#808080',
                           '#800080', '#FFD700', "#B1BBB8", "#F3E6C3"])
        np.random.shuffle(colors)
        # colors = np.linspace(0, 1, len(self.classifiers))

        # plt.scatter(X[:,0], X[:,1], color=colors[predictions])

        # for cluster in self.boundaries.clusters:
        #     draw_cluster(cluster)

        # for classifier in self.classifiers:
        #     draw_classifier(classifier)

        # for classifier in self.boundaries.classifiers:
        #     draw_classifier(classifier)

        for i in range(len(self.clusters)):
            draw_cluster(X[self.clusters[i]], color=colors[i])
            # draw_classifier(self.classifiers[i], color=colors[i])
            # if i == len(self.classifiers) - 1:
            #     continue
            # draw_classifier(self.boundaries.boundaries[i], color=colors[i])
        plt.savefig(f"./figures/{name}_cluster.pdf", format='pdf', dpi=600)
        plt.show()


def main(name, num, data_dir, threshold, penalty_term, gap):
    print("Name:", name)
    # Example settings
    # n_samples = 300
    # outliers_fraction = 0.15
    # n_outliers = int(outliers_fraction * n_samples)
    # n_inliers = n_samples - n_outliers

    # build dataset
    # blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
    # X = make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5, **blobs_params)[0]
    # # Add outliers
    # rng = np.random.RandomState(42)
    # X = np.concatenate([X, rng.uniform(low=-6, high=6,
    #                                    size=(n_outliers, 2))], axis=0)
    #
    # y = np.concatenate([np.zeros(n_inliers), np.ones(n_outliers)])

    # plt.plot(X, y, color=color)
    if use_lof:
        X, y, lof_krange, N_range, knn_krange, if_range, mahalanobis_N_range = load_data(name,
                                                                                         data_dir=data_dir)
        X, y = np.array(X), np.array(y)
        # get the noisy labels
        def run_lof(X, y, num_outliers=560, k=60):
            clf = LocalOutlierFactor(n_neighbors=k)
            clf.fit(X)
            lof_scores = -clf.negative_outlier_factor_
            threshold = np.sort(lof_scores)[::-1][num_outliers]
            lof_predictions = np.array(lof_scores > threshold)
            lof_predictions = np.array([int(i) for i in lof_predictions])
            print("F-1 score of LOF outlier detection:", metrics.f1_score(y, lof_predictions))
            return lof_predictions, lof_scores

        lof_predictions, lof_scores = run_lof(X, y, k=60, num_outliers=int(np.sum(y)))

        transformer = RobustScaler().fit(X)
        X_transformed = transformer.transform(X)
        local_lime = LocalDT(num=num)

        # for Annthyroid, do not use robustscaler
        # number, length, n_c = local_lime.run(X, lof_predictions, threshold, penalty_term, gap=gap)
        # for others
        number, length, n_c = local_lime.run(X_transformed, lof_predictions, threshold, penalty_term, gap=gap)

        # local_lime.draw(X)
        # draw clusters
        # local_lime.draw(X)
        # np.save(f"./{name}.npy", np.array(local_lime.clusters))
        # X_embed = TSNE(n_components=2).fit_transform(X)
        # local_lime.draw(X_embed)
        return number, length, n_c, local_lime.clusters

    if use_misdetect:
        data_dir = r'/home/jinruochen/outlier_summ/cleaned_data/'
        data_dir = r'../cleaned_data/'
        X, y = load_misdetect_prediction(name=name, data_dir=data_dir) # load misdetect result
        X, y = np.array(X), np.array(y)
        misdetect_predictions = y
        transformer = RobustScaler().fit(X)
        X_transformed = transformer.transform(X)

        local_lime = LocalDT(num=num)

        number, length, n_c = local_lime.run(X_transformed, misdetect_predictions, threshold, penalty_term = penalty_term, gap=gap)
        # np.save(f"./{name}.npy", np.array(local_lime.clusters))

        # X_embed = TSNE(n_components=2).fit_transform(X)
        # local_lime.draw(X_embed)
        return number, length, n_c, local_lime.clusters


if __name__ == '__main__':

    if len(sys.argv) > 1:
        if sys.argv[1] == 'All':
            datasets = ['Pendigits', 'Pima', "Mammography", "Satimage-2", "PageBlock", "satellite", "ALOI", "optdigits", "Annthyroid", "Thursday"]
        else:
            datasets = [sys.argv[1]]
    else:
        datasets = ['Pendigits']

    # max_lengths = [2, 4, 6, 8, 10, 12]
    max_lengths = [10] # maximum rule length
    numbers = [2,4,8] # number of initial clusters
    # numbers = [2]
    # thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    thresholds = [0.8]
    penalty_terms = [0.0001]
    use_lof = 1
    use_misdetect = 0

    # df_f1 = pd.DataFrame(columns=names, index=thresholds)
    results = []

    for name in datasets:
        best_length = 1e9
        for num in numbers:
            for threshold in thresholds:
                for ml in max_lengths:
                    for penalty_term in penalty_terms:
                        print(penalty_term)

                        gap = load_default_gap(name=name, num=num)

                        max_length_global = ml

                        if use_lof:
                            print('running l-stair on noisy data')
                            print('F1 threshold', threshold)
                            data_dir = r"../data/"
                            number, length, n_c, clusters = main(name, num=num, data_dir=data_dir, threshold=threshold, penalty_term=penalty_term, gap=gap)
                            results.append([name, num, threshold, ml, number, length, n_c])

                        if use_misdetect:
                            print('running l-stair on cleaned data')
                            data_dir = r'/data/jinruochen/outlier_summ/cleaned_data' # run on remote server
                            data_dir = r'../cleaned_data/' # run on local server
                            number, length, n_c, clusters = main(name, num=num, data_dir=data_dir, threshold=threshold, penalty_term=penalty_term, gap=gap)
                            results.append([name, num, threshold, ml, number, length, n_c])

                        # save_dir = './L-STAIR/'
                        # if not os.path.exists(save_dir):
                        #     os.makedirs(save_dir)

                        if length < best_length and threshold == 0.8:
                            # np.save(f"./L-STAIR/{name}_{threshold}_{num}.npy", np.array(clusters))
                            best_length = length
                            print(f"the best length is {best_length}")
                        print(pd.DataFrame(results, columns=['name', 'n', 'threshold', 'max_length', 'rule_number', 'rule_length', 'cluster_num']))

    results = pd.DataFrame(results, columns=['name', 'n', 'threshold', 'max_length', 'rule_number', 'rule_length', 'cluster_num'])
    print('saving result...')
    results.to_csv("effect_f1_lstair.csv", index=False)
