from pathlib import Path
import os.path
import pandas as pd
from numpy import nan
from typing import Sequence
from dataclasses import dataclass
import numpy as np
from ast import literal_eval

train_data_folder = Path(__file__).parent.joinpath('dataset/train')
test_data_folder  = Path(__file__).parent.joinpath('dataset/test')
data_folder       = Path(__file__).parent.joinpath('dataset')
anomalies_name = 'labeled_anomalies'


def is_smap(dataset_name):
    filename_csv = data_folder.joinpath(f"{anomalies_name}.csv")
    if os.path.isfile(filename_csv):
        dataset = pd.read_csv(filename_csv)
        dataset = dataset[dataset['chan_id'] == dataset_name]
        print()
        return True if dataset['spacecraft'].any() == 'SMAP' else False

@dataclass
class ReadData:
    
    def __init__(self, dataset_name, spacecraft='SMAP', verbose=False):
        self.dataset_name = dataset_name
        self.spacecraft = spacecraft
        self.verbose=verbose
    
    def load_data(self):
        
        def read_set(filename):
            if os.path.isfile(filename):
                print("Reading: ", filename)
                dataset = np.load(filename)
                #dataset = self.drop_missing(dataset)
                # summarize
                if self.verbose:
                    print(dataset.shape)
                return dataset
            else:
                print('{} does not exist'.format(filename))
        
        train_name = train_data_folder.joinpath(f"{self.dataset_name}.npy")
        test_name  = test_data_folder.joinpath(f"{self.dataset_name}.npy")
        return read_set(train_name), read_set(test_name)
    
    def load_anomalies(self):
        filename_csv = data_folder.joinpath(f"{anomalies_name}.csv")
        if os.path.isfile(filename_csv):
            if self.verbose:
                print("Reading: ", filename_csv)
            dataset = pd.read_csv(filename_csv)
            dataset = self.drop_missing(dataset)
            # summarize
            if self.verbose:
                print(dataset.shape)
        dataset = dataset[dataset['chan_id'] == self.dataset_name]
        return np.array(literal_eval(dataset['anomaly_sequences'].to_numpy()[0]))
        
    
    def drop_missing(self,dataset):
        # checking missing values
        dataset = dataset.dropna() 
        # dropping missing valies
        if self.verbose:
            print('....Dropped Missing value row....')
            print('Rechecking Missing values:', dataset.isnull().sum()) 
        # checking missing values
        return dataset;
  