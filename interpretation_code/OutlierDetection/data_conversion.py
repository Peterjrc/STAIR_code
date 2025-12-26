import numpy as np
import pandas as pd
from numpy import cross
from scipy.io import arff, loadmat
import os
import csv
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.model_selection import cross_validate


# Load .arff file

def load_arff(data_dir, file_name):
    data_path = os.path.join(data_dir, file_name)
    data, meta = arff.loadarff(data_path)
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # print(df.head().iloc[:,-1])
    df.iloc[:,-1] = df.iloc[:,-1].apply(lambda x: 1 if x == b'yes' else 0)
    if 'id' in df.columns:
        del df['id']

    metadata_lines = []
    for attr_name, attr_type in zip(meta.names(), meta.types()):
        if attr_name == 'id':
            continue
        if attr_type == 'numeric':
            attr_type = 'continuous'
        elif attr_type == 'nominal':
            attr_type = 'discrete'
        metadata_lines.append(f"{attr_name} {attr_type}")

    metadata_lines.append("LABEL_POS -1")
    output_path_info = os.path.join(save_dir, f"{file_name.split('.')[0]}.info")

    with open(output_path_info, mode="w") as info_file:
        info_file.write("\n".join(metadata_lines))

    output_path_data = os.path.join(save_dir, f'{file_name.split(".")[0]}.data')
    # Save to .data file
    df.to_csv(output_path_data, index=False, header=False, sep=',')
    print(f"Info file saved to {output_path_info}")
    print(f"Data file saved to {output_path_data}")

# load_arff(data_dir, 'Pima_withoutdupl_norm_35.arff')


def load_mat(data_dir, file_name):
    # Load the .mat file
    data_path = os.path.join(data_dir, file_name)
    mat_data = loadmat(data_path)
    X, y = mat_data['X'], mat_data['y']
    # y = [item[0] for item in mat_data['y']]
    raw_data = np.concatenate((X,y),axis=1)
    df = pd.DataFrame(raw_data)
    if 'id' in df.columns:
        del df['id']
    df.columns = [ ''.join(('attr',str(x))) if x in df.columns[:-1] else 'outlier' for x in df.columns]
    # print(df.columns)
    info_dict = {}
    info_lines = []

    if file_name == 'shuttle.mat':
        for feature in df.columns:
            if feature == 'outlier':
                info_dict[feature] = 'discrete'
            else:
                info_dict[feature] = 'continuous'

        for feature, type in info_dict.items():
            info_lines.append(f"{feature} {type}")
        info_lines.append("LABEL_POS -1")

    elif file_name == 'mammography.mat':
        for feature in df.columns:
            if feature == 'outlier':
                info_dict[feature] = 'discrete'
            else:
                info_dict[feature] = 'continuous'

        for feature, type in info_dict.items():
            info_lines.append(f"{feature} {type}")
        info_lines.append("LABEL_POS -1")

    elif file_name == 'shuttle.mat':
        for feature in df.columns:
            if feature == 'outlier':
                info_dict[feature] = 'discrete'
            else:
                info_dict[feature] = 'continuous'

        for feature, type in info_dict.items():
            info_lines.append(f"{feature} {type}")
        info_lines.append("LABEL_POS -1")

    elif file_name == 'cover.mat':
        for feature in df.columns:
            if feature == 'outlier':
                info_dict[feature] = 'discrete'
            else:
                info_dict[feature] = 'continuous'

        for feature, type in info_dict.items():
            info_lines.append(f"{feature} {type}")
        info_lines.append("LABEL_POS -1")

    elif file_name == 'satellite.mat':
        for feature in df.columns:
            if feature == 'outlier':
                info_dict[feature] = 'discrete'
            else:
                info_dict[feature] = 'continuous'

        for feature, type in info_dict.items():
            info_lines.append(f"{feature} {type}")
        info_lines.append("LABEL_POS -1")

    elif file_name == 'satimage-2.mat':
        for feature in df.columns:
            if feature == 'outlier':
                info_dict[feature] = 'discrete'
            else:
                info_dict[feature] = 'continuous'

        for feature, type in info_dict.items():
            info_lines.append(f"{feature} {type}")
        info_lines.append("LABEL_POS -1")

    elif file_name == 'pendigits.mat':
        for feature in df.columns:
            if feature == 'outlier':
                info_dict[feature] = 'discrete'
            else:
                info_dict[feature] = 'continuous'

        for feature, type in info_dict.items():
            info_lines.append(f"{feature} {type}")
        info_lines.append("LABEL_POS -1")

    output_path_data = os.path.join(save_dir, f'{file_name.split(".")[0]}.data')
    output_path_info = os.path.join(save_dir, f"{file_name.split('.')[0]}.info")

    with open(output_path_info, mode="w") as info_file:
        info_file.write("\n".join(info_lines))

    # Save to .data file
    df.to_csv(output_path_data, index=False, header=False, sep=',')

    # print(f"Info file saved to {output_path_info}")
    print(f"Data file saved to {output_path_data}")
    print(f"Info file saved to {output_path_info}")


# a = load_mat(data_dir, 'pendigits.mat')

def load_csv(data_dir, file_name):
    data_path = os.path.join(data_dir, file_name)
    if file_name == 'winequality-white.csv':
        df = pd.read_csv(data_path, sep = ';')

    elif file_name == 'Thursday-2018_processed.csv':
        df = pd.read_csv(data_path)

    output_path_data = os.path.join(save_dir, f'{file_name.split(".")[0]}.data')
    output_path_info = os.path.join(save_dir, f"{file_name.split('.')[0]}.info")
    info_lines = []

    if file_name == 'winequality-white.csv':
        for feature in df.columns:
            # print(feature)
            if feature != 'quality':
                info_lines.append(f"{feature.replace(' ', '')} {'continuous'}")
            else:
                info_lines.append(f"{feature.replace(' ', '')} {'discrete'}")

        info_lines.append("LABEL_POS -1")

    if file_name == 'Thursday-2018_processed.csv':
        discrete_feature = ["Fwd PSH Flags", "Fwd URG Flags", \
                            "Fwd Header Len", "Bwd Header Len",\
                            "FIN Flag Cnt", \
                            "SYN Flag Cnt", \
                            "RST Flag Cnt", \
                            "PSH Flag Cnt", \
                            "ACK Flag Cnt",\
                            "URG Flag Cnt", \
                            "CWE Flag Count", \
                            "ECE Flag Cnt", \
                            "Fwd Act Data Pkts",\
                            "Fwd Seg Size Min", 'label']
        for feature in df.columns:
            if feature == 'id':
                # eliminate id column
                del df[feature]
                continue
            if feature in discrete_feature:
                info_lines.append(f"{feature.replace(' ', '')} {'discrete'}")
            else:
                info_lines.append(f"{feature.replace(' ', '')} {'continuous'}")
        info_lines.append("LABEL_POS -1")
        # print(len(df.columns))

    with open(output_path_info, mode="w") as info_file:
        info_file.write("\n".join(info_lines))
    # Save to .data file
    df.to_csv(output_path_data, index=False, header=False, sep=',')
    # print(f"Info file saved to {output_path_info}")
    print(f"Data file saved to {output_path_data}")
    print(f"Info file saved to {output_path_info}")

# load_csv(data_dir, 'winequality-white.csv')
# a  = load_csv(data_dir, 'Thursday-01-03-2018_processed.csv')

# mat_data = pd.DataFrame(loadmat('/Users/mse540/Desktop/Research/outlier_summ_复现/data/satellite.mat')['X']).to_csv('satellite_data.csv', index = False)

# run lof

def run_lof(X, y , file_name, num_outliers=560, k=60):

    if file_name == 'PageBlocks_norm_10.data':
        lof_krange = list(range(10, 110, 10)) * 6

    elif file_name == 'Pima_withoutdupl_norm_35.data':
        lof_krange = list(range(10, 210, 10)) * 6

    elif file_name == 'shuttle.data':
        lof_krange = list(range(10, 110, 10)) * 6

    elif file_name == 'satimage-2.data':
        lof_krange = list(range(10, 110, 10)) * 6

    elif file_name == 'pendigits.data':
        lof_krange = list(range(10, 110, 10)) * 6

    elif file_name == 'mammography.data':
        lof_krange = list(range(10, 110, 10)) * 6

    elif file_name == 'satellite.data':
        lof_krange = list(range(10, 110, 10)) * 6

    elif file_name == 'cover.data':
        lof_krange = list(range(10, 110, 10)) * 6

    elif file_name == 'Thursday-2018_processed.data':
        lof_krange = list(range(10, 100, 10)) * 6

    elif file_name == 'SpamBase_withoutdupl_norm_40.data':
        lof_krange = list(range(10, 110, 10)) * 6

    elif file_name == 'winequality-white.data':
        pass

    clf = LocalOutlierFactor(n_neighbors=lof_krange[7])

    clf.fit(X)
    lof_scores = -clf.negative_outlier_factor_
    threshold = np.sort(lof_scores)[::-1][num_outliers]
    lof_predictions = np.array(lof_scores > threshold)
    lof_predictions = np.array([int(i) for i in lof_predictions])
    f1_score = metrics.f1_score(y, lof_predictions)
    # print(f"F-1 score of LOF outlier detection for {file_name}:", f1_score)
    return lof_predictions, lof_scores, f1_score


# df = np.loadtxt("./dataset/cover.data", delimiter = ',')

# result, score = run_lof(df[:,:-1], df[:,-1])

if __name__ == '__main__':
    data_dir = '/Users/mse540/Desktop/Research/outlier_summ_复现/data/'
    save_dir = '/Users/mse540/Desktop/Research/rrl/dataset/'
    ratio = 1.0
    convert_data = True

    if convert_data:
        data_list = os.listdir(data_dir)
        lof = True
        for file in data_list:

            if file == 'winequality-white.csv':
                continue

            if file.split('.')[1] == 'arff':
                load_arff(data_dir, file)

            elif file.split('.')[1] == 'mat':
                load_mat(data_dir, file)

            elif file.split('.')[1] == 'csv':
                load_csv(data_dir, file)

            if lof:
                df = np.loadtxt(f"{save_dir}{file.split('.')[0]}.data", delimiter=',')
                X, y = df[:,:-1], df[:,-1]
                lof_prediction, lof_score, f1_score = run_lof(X, y, num_outliers=int(sum(y)), file_name=f"{file.split('.')[0]}.data")
                outlier_idxes = np.where(lof_prediction == 1)[0]
                # Outliers = X[outlier_idxes]

                idxes = np.where(lof_prediction == 0)[0]
                # Inliers = X[idxes[selected_indices]]

                # X_new = np.concatenate([Outliers, Inliers], axis=0)
                # lof_prediction_new = np.concatenate([np.ones(len(Outliers)), np.zeros(len(Inliers))], axis=0)
                X_df = pd.DataFrame(X, columns=[f'attr{i}' for i in range(1, X.shape[1]+1)])
                y[outlier_idxes] = 1
                y[idxes] = 0
                lof_prediction_df = pd.DataFrame(y, columns=['label'])
                assert len(X_df) == len(lof_prediction_df)
                print(X_df.shape, lof_prediction_df.shape)

                # # convert back to .data file
                # X_new_df = pd.DataFrame(X, columns=[f'attr{i}' for i in range(1, X.shape[1]+1)])
                # lof_prediction_new_df = pd.DataFrame(lof_prediction_new, columns=['label'])
                df_new = pd.concat((X_df, lof_prediction_df), axis=1)
                # output_path_data = os.path.join(save_dir, f'{file.split(".")[0]}.data')
                # data = np.savetxt(output_path_data, df_new, delimiter=',')
                if not os.path.exists("/Users/mse540/Desktop/Research/outlier_summ_复现/lof_predictions"):
                    os.mkdir("/Users/mse540/Desktop/Research/outlier_summ_复现/lof_predictions")
                predictions_dir = "/Users/mse540/Desktop/Research/outlier_summ_复现/lof_predictions/"

                df_new.to_csv(f"{predictions_dir}{file.split('.')[0]}.csv")
                print(f"{file} saved successfully at {predictions_dir}")

    # data_dir = '/Users/mse540/Desktop/Research/outlier_summ_复现/data_lof'
    # data_list = os.listdir(data_dir)
    # result = pd.DataFrame()
    # for data in data_list:
    #     # print(f"{data_dir}{data}")
    #     df = np.loadtxt(f"{data_dir}{data}", delimiter=',')
    #     X, y = df[:, :-1], df[:, -1]
    #     lof_prediction, lof_score, f1_score = run_lof(X, y, num_outliers=int(sum(y)), file_name=f"{data}")
    #     result.loc[data, 'f1_score'] = f1_score
    #     assert len(X) == len(lof_prediction)
    # print(result)