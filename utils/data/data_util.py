from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np


def describe(dataset):
    print(dataset.describe().transpose());


def normalize(dataset):
    return dataset;


def vis_behaviour(dataset, column_name):
    pyplot.figure(figsize=(30,12))
    # Generate Histogram
    ax = pyplot.subplot(2, 2, 1)
    ax.hist(dataset[column_name])
    pyplot.title("freq_histogram", y=0.05,x=0.9, loc='right')

    # Generate Boxplot
    ax = pyplot.subplot(2, 2, 2)
    ax.boxplot(dataset[column_name])
    pyplot.title("box_plot", y=0.05,x=0.9, loc='right')

    pyplot.show()


def split(dataset, column_name, scaler, test_size=0.05, scale_full_set=False):
    train, test = train_test_split(dataset, test_size=test_size, random_state=42, shuffle=False);
    print("Train_shape: ", train.shape);
    print("Test_shape: ", test.shape);

    # data standardization
    print("Data Standardization with ", str(scaler))
    #scaler = preprocessing.StandardScaler()
    #scaler = preprocessing.RobustScaler(quantile_range=(25, 75))
    scaler = scaler.fit(train[[column_name]])
    train[column_name] = scaler.transform(train[[column_name]])
    test[column_name] = scaler.transform(test[[column_name]])

    if scale_full_set:
        full_set = dataset.copy()
        full_set[column_name] = scaler.transform(full_set[[column_name]])
        return (train, test, full_set)

    return (train,test)


def split_multivariate(dataset, scaler, test_size=0.05, timestamp_label='timestamp'):
    train, test = train_test_split(dataset, test_size=test_size, random_state=42, shuffle=False);
    print("Train_shape: ", train.shape);
    print("Test_shape: ", test.shape);

    # data standardization
    print("Data Standardization with ", str(scaler))
    for column_name in dataset.columns:
        if column_name != timestamp_label:
            scaler = scaler.fit(train[[column_name]])
            train[column_name] = scaler.transform(train[[column_name]])
            test[column_name] = scaler.transform(test[[column_name]])

    return train, test


def create_dataset(X, y, time_steps=1, timestamp_label='timestamp'):
    X = X.drop(columns=[timestamp_label])
    a, b = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        a.append(v)
        b.append(y.iloc[i + time_steps])
    return np.array(a), np.array(b)


def create_dataset_np(X, time_steps=1):
    a = []
    for i in range(len(X) - time_steps):
        v = X[i:i+time_steps]
        a.append(v)
    return np.array(a)


def data_standardization(train, test, scaler):
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test
