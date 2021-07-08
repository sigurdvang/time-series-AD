from pathlib import Path
import os.path
import pandas as pd
from numpy import nan
from typing import Sequence
from dataclasses import dataclass
import numpy as np
from ast import literal_eval

train_data_folder     = Path(__file__).parent.joinpath('dataset/train')
test_data_folder      = Path(__file__).parent.joinpath('dataset/test')
data_folder           = Path(__file__).parent.joinpath('dataset')
anomaly_labels_folder = Path(__file__).parent.joinpath('dataset/labels')

@dataclass
class ReadData:
    
    def __init__(self, group_index=1, index=1):
        self.machine_name = 'machine-{}-{}'.format(group_index, index)
        
    
    def load_data(self):
        
        def read_set(filename):
            if os.path.isfile(filename):
                print("Reading: ", filename)
                dataset = pd.read_csv(filename, header=None)
                # summarize
                print(dataset.shape)
                return dataset
            else:
                print('{} does not exist'.format(filename))
        
        train_name = train_data_folder.joinpath(f"{self.machine_name}.txt")
        test_name  = test_data_folder.joinpath(f"{self.machine_name}.txt")
        return read_set(train_name), read_set(test_name)
    
    def load_anomalies(self):
        filename = anomaly_labels_folder.joinpath(f"{self.machine_name}.txt")
        if os.path.isfile(filename):
            print("Reading: ", filename)
            anomaly_labels = pd.read_csv(filename, header=None)
            #dataset = self.drop_missing(dataset)
        
        col = anomaly_labels.columns[0]

        anomaly_segments = {
            'start': [],
            'end': [],
        }

        in_segment=False
        for i in range(len(anomaly_labels[col])):
            is_anomaly_label = anomaly_labels[col][i] == 1
            if is_anomaly_label and not in_segment:
                in_segment=True
                anomaly_segments['start'].append(i)
            elif not is_anomaly_label and in_segment:
                in_segment=False
                anomaly_segments['end'].append(i-1)
        
        return pd.DataFrame.from_dict(anomaly_segments)
    
    def drop_missing(self,dataset):
        # checking missing values
        dataset = dataset.dropna() 
        # dropping missing valies
        print('....Dropped Missing value row....')
        print('Rechecking Missing values:', dataset.isnull().sum()) 
        # checking missing values
        return dataset;
  