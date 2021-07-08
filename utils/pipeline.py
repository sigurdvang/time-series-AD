from utils.data.smd import data_reader as smd_data_reader
from sklearn import preprocessing
import numpy as np
from utils.evaluation import find_epsilon, partition_anomalous_subsequences, calc_pointwise_metrics
import pandas as pd
from pandas.api.types import is_numeric_dtype
import random
from utils.data.nasa import data_reader as nasa_data_reader
from utils.data import data_util


def scale_data(dataset):
    """
    Minmax scales dataset into range (-1, 1)
    args:
        dataset
    returns:
        scaled dataset
    """
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(dataset)


def smd_remove_dead_signals(dataset, i, j):
    """
    For the SMD datasets, channels that are static througout time-series is removed.
    args:
        dataset
        i: group_index of dataset
        j: index of dataset
    returns:
        dataset with removed dead channels
    """
    if [i, j] in [[1,2], [2,1], [3,2], [3,7]]:
        dead_signals = [7, 26, 28, 36, 37]
    elif [i, j] in [[1,3], [1,4], [1,7], [2, 8]]:
        dead_signals = [7, 26, 36, 37]
    elif [i, j] in [[1, 5], [2, 6], [2, 9], [3, 4]]:
        dead_signals = [4, 7, 16, 17, 26, 28, 36, 37]
    elif [i, j] in [[1, 1], [3, 9]]:
        dead_signals = [4, 7, 16, 17, 36, 37]
    elif [i, j] in [[1,8], [2, 4]]:
        dead_signals = [7, 16, 17, 26, 28, 36, 37]
    elif [i, j] in [[3, 6], [3, 8]]:
        dead_signals = [4, 7, 16, 17, 26, 36, 37]
    elif i == 1 and j == 6:
        dead_signals = [7, 26, 28, 37]
    elif i == 2 and j == 2:
        dead_signals = [7, 16, 26, 28, 36, 37]
    elif i == 2 and j == 3:
        dead_signals = [7, 16, 17, 36, 37]
    elif i == 2 and j == 5:
        dead_signals = [7, 16, 17, 26, 37]
    elif i == 2 and j == 7:
        dead_signals = [7, 16, 17, 26, 36, 37]
    elif i == 3 and j == 1:
        dead_signals = [7, 16, 17, 28, 36, 37]
    elif i == 3 and j == 3:
        dead_signals = [7, 28]
    elif i == 3 and j == 5:
        dead_signals = [4, 7, 26, 34, 35, 36, 37]
    else:
        raise ValueError
    return pd.DataFrame(np.delete(dataset.values, obj=dead_signals, axis=1))


def get_rolling_window_data(n_steps, train, test):
    """
    Partitions datasets on the form (length, n_features) to a series of subsequences of length
    n_steps on the form (length, n_steps, n_features) using a rolling window technique.
    args:
        n_steps: length of subsequences
        train: training set
        test: testing set
    returns:
        X_train: converted training set
        x_test: converted test set
        
    """
    X_train = data_util.create_dataset_np(train, n_steps)
    X_test = data_util.create_dataset_np(test, n_steps)
    return X_train, X_test


class Pipeline:
    """
    Performs the anomaly pipeline.
    1. Reads and pre_processes data.
    2. Scores anomaly data
    3. Finds anomalies from anomaly scores

    Also has functionality to evaluate the quality of found anomalies
    To work with custom data, one can subclass this class and override the load data method
    args:
        anomaly_scoring_model: model to be used for anomaly scoring
        dataset_name: dataset to be used 
        dataset_args: arguments required for data loading 
        n_steps: lenght of sub sequences 
        batch_size: batch size
        contamination_level: the amount of contamination that should be added to data. Used for
                             contamination experiments
    """

    def __init__(self, anomaly_scoring_model, dataset_name, dataset_args, n_steps, batch_size, contamination_level=0):
        train, test, known_anomalies = self.load_data(dataset_name, dataset_args)
        self.train = self.contaminate_data(train, known_anomalies, contamination_level) if contamination_level > 0 else train
        self.test = test
        print('sjape', self.train.shape)

        self.X_train, self.X_test = get_rolling_window_data(n_steps, train, test)

        self.known_anomalies = known_anomalies
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.as_model = anomaly_scoring_model

    def load_data(self, dataset, args):
        """
        Loads data 
        """
        if dataset == 'smd':
            i, j = args['group_index'], args['index']
            reader = smd_data_reader.ReadData(i, j)
            train, test = reader.load_data()
            train, test = smd_remove_dead_signals(train, i, j), smd_remove_dead_signals(test, i, j)

            # get anomalies
            known_anomalies = reader.load_anomalies()
            known_anomalies = pd.DataFrame(known_anomalies, columns=['start', 'end'])
        if dataset == 'nasa':
            file_name = args['file_name']
            print(file_name)
            reader = nasa_data_reader.ReadData(file_name)
            train, test = reader.load_data()
            train = np.delete(train, obj=[i for i in range(1, train.shape[1])], axis=1)
            test = np.delete(test, obj=[i for i in range(1, test.shape[1])], axis=1)
            known_anomalies = reader.load_anomalies()
            df = {'start': [], 'end': []}
            for a in known_anomalies:
                df['start'].append(a[0])
                df['end'].append(a[1])
            known_anomalies = pd.DataFrame(df)

        # scale
        train, test = scale_data(train), scale_data(test)
        return train, test, known_anomalies

    def calc_anomaly_score(self):
        """
        Gets anomaly score over test dataset
        """
        return self.as_model.point_wise_anomaly_score(X_windows=self.X_test, values=self.test, window_size=self.n_steps)

    def find_anomalies(self, anomaly_scores):
        """
        Find anomalies given anomaly scores
        """
        smoothing_window = int(self.batch_size * self.n_steps * 0.05)
        smoothed_scores = pd.DataFrame(anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()
        threshold = find_epsilon(smoothed_scores)
        predictions, labels = partition_anomalous_subsequences(smoothed_scores, self.known_anomalies, threshold, advance=1)
        return predictions, labels

    def calc_metrics(self, predictions, labels):
        """
        calculates metrics given found anomalies and ground truth
        """
        f1_score, precision, recall, TP, TN, FP, FN = calc_pointwise_metrics(predictions, labels)
        return f1_score, precision, recall, TP, TN, FP, FN

    def predict_and_evaluate(self):
        """
        Aggregate method that performs both anomaly scoring and evaluation in one
        """
        anomaly_scores = self.calc_anomaly_score()
        anomalies, labels = self.find_anomalies(anomaly_scores)
        return self.calc_metrics(anomalies, labels)

    def contaminate_data(self, data, known_anomalies, contamination_level):
        """
        Method that contaminates data by adding synthetic anomalies to it
        args:
            data: training dataset
            known_anomalies: ground truth anomalies that are in the test set. Used to generate new ones
            contamination level: level to which one should contaminate data.
        """
        def add_noise(d, s_index, e_index):
            n_features = data.shape[-1]
            for j in range(s_index, e_index + 1):
                noise = np.random.uniform(low=0.7, high=1.3, size=(n_features,))
                if type(d) == np.ndarray:
                    d[j] *= noise
                else:
                    d.values[j] *= noise
            return d

        n_anomalies = 5 * contamination_level
        max_seq_len = float("-inf")
        min_seq_len = float("inf")
        # find max and minimum anomaly sequence length
        # the new anomalies will be in this range
        for i in range(len(known_anomalies['start'])):
            diff = abs(known_anomalies['start'][i] - known_anomalies['end'][i])
            if diff > max_seq_len:
                max_seq_len = diff
            if diff < min_seq_len:
                min_seq_len = diff
        # Creates n new anomalies at random
        for i in range(n_anomalies):
            start = random.randint(0, data.shape[0] - max_seq_len + 1)
            end = start + random.randint(min_seq_len, max_seq_len)
            # add noise to the new anomaly region
            data = add_noise(data, start, end)
        return data


class TelenorPipeline(Pipeline):
    """
    Pipeline to work with Telenor data used in thesis.

    Example of how the AD pipeline can be extended to work with new datasets, by subclassing Pipeline
    and adding new load_data method, to override parent.

    As the Telenor dataset is not open, the contents of the methods are retracted, however, the
    subclass still serves as an example of how one can fit the pipeline to ones own datasets
    """

    def __init__(self, anomaly_scoring_model, dataset_name, dataset_args, n_steps, batch_size):
        super().__init__(anomaly_scoring_model, dataset_name, dataset_args, n_steps, batch_size)

    def fill_nan(self, p_df, nan_cols):
        pass

    def load_data(self, dataset, args):
        pass

    def remove_cols(self, p_df):
        pass

    def remove_labels(self, p_df):
        pass


