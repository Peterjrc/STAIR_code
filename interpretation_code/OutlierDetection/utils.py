import os

import numpy as np
from scipy.io import arff
import pandas as pd
from warnings import simplefilter
# from load_data import load_rrl_prediction
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# from load_data import load_data

simplefilter(action='ignore', category=FutureWarning)

view_rrl_prediction = False

def load_dataset(filename):
    with open(filename, 'r') as f:
        data, meta = arff.loadarff(f)
    data = pd.DataFrame(data)
    if filename != '../data/Mulcross.arff':
        X = data.drop(columns=['id', 'outlier'])
        y = data["outlier"].map(lambda x: 1 if x == b'yes' else 0).values
    else:
        X = data.drop(columns=['Target'])
    # Map dataframe to encode values and put values into a numpy array
        y = data["Target"].map(lambda x: 1 if x == b'Anomaly' else 0).values
    return X, y


def create_folder_rrl_data(data_dir = '/Users/mse540/Desktop/Research/outlier_summ_复现/data_rrl/'):
    data_list = ['Pendigits', 'PageBlock', "Shuttle", "Pima", "Mammography", "Satimage-2", "cover", "satellite",
                     "SpamBase", "Thursday"]
    if not os.path.exists(data_dir): # create folder for rrl data
        os.makedirs(data_dir)
    for data in data_list:
        if not os.path.exists(data_dir + data):
            os.makedirs(data_dir + data)


def load_misdetect_prediction(name, data_dir='../cleaned_data/'):
    # load prediction results from rrl model
    path = os.path.join(data_dir, f'{name}_cleaned.csv')
    data = pd.read_csv(path)
    # print(data.head())
    y = data.iloc[:,-1].values
    X = data.iloc[:,:-1]
    print(np.shape(X), np.shape(y))
    print(sum(y))
    return X, y

# def read_csv(name, data_dir='../data_raw/'):
#     data = pd.read_csv(f"{data_dir}{name}.csv")


def view_data(file_name=None, data=None):

    if view_rrl_prediction:
        X, y = load_rrl_prediction(file_name)

    else:
        X, y = data[:,:-1], data[:,-1]

    tsne = TSNE(n_components=2, perplexity=30, random_state=0)

    # Fit and transform the data
    X_2d = tsne.fit_transform(X)
    # Create a scatter plot
    plt.figure(figsize=(8, 6))

    # Color points by their labels (if available)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)

    # Add labels and title
    plt.colorbar(label="Labels")
    plt.title(f"t-SNE Visualization of {file_name} (lof predictions)")
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")

    # Show the plot
    plt.show()


def load_npz(path):
    filepath = path
    data = np.load(filepath)

    X = data['X']
    # from sklearn.preprocessing import StandardScaler
    # enc = StandardScaler()
    # x = enc.fit_transform(x)
    # print(x.shape)
    y = data['y']
    # print(y.shape)
    return X,y



def get_predictions_scores(scores, num_outliers=400, method_name='LOF'):
    threshold = np.sort(scores)[::-1][num_outliers]
    # threshold, max_f1 = get_best_f1_score(y, lof_scores)
    predictions = np.array(scores > threshold)
    predictions = np.array([int(i) for i in predictions])
    #     print('F1 for {} : {}'.format(method_name, metrics.f1_score(y, predictions)))
    return predictions, scores




