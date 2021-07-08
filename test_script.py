from models.build_model import build_beatgan, build_reggan, build_tadgan, build_madgan
from models.sklearn_wrappers import KNN, GMM
from models.torch_lstm_forecaster import LSTMForecaster
import torch
import sys
from utils.data.nasa.data_reader import is_smap
import numpy as np
from utils.pipeline import Pipeline
import configparser


"""
This script runs tests on various models on choosen datasets. To perform new test with new models
on new datasets, this script can serve as an example
"""


def convert_string_to_list(l, d_type):
    """
    Takes string and converts it to a list of a given datatype
    args:
        l: string to be convert to list
        d_type: data type to use for conversion
    retursn:
        list of type d_type
    """
    l = l.split(',')
    for i, el in enumerate(l):
        l[i] = d_type(el)
    return l


class Config:
    """
    Simple class that retrieves models and system configurations used to perform tests
    """

    def __init__(self, config_file_name):
        config = configparser.ConfigParser()
        config.read(config_file_name)

        self.MODEL_TYPE = config['SYSTEM']['MODEL_TYPE']
        self.DATASET = config['SYSTEM']['DATASET']
        self.n_steps = int(config['SYSTEM']['n_steps'])
        self.batch_size = int(config['SYSTEM']['batch_size'])
        self.n_epochs = int(config['SYSTEM']['n_epochs'])
        self.contamination_level = int(config['SYSTEM']['contamination_level'])

        if self.MODEL_TYPE == 'BeatGAN':
            self.filters = int(config['BeatGAN']['filters'])
            self.kernel_size = int(config['BeatGAN']['kernel_size'])
            self.z_dim = int(config['BeatGAN']['z_dim'])
            self.reg_w = float(config['BeatGAN']['reg_w'])
            self.g_lr = float(config['BeatGAN']['g_lr'])
            self.d_lr = float(config['BeatGAN']['d_lr'])
        if self.MODEL_TYPE == 'MAD-GAN':
            self.g_lr = float(config['MAD-GAN']['g_lr'])
            self.d_lr = float(config['MAD-GAN']['d_lr'])
            self.z_dim = int(config['MAD-GAN']['z_dim'])
            self.n_layers = int(config['MAD-GAN']['n_layers'])
            self.hidden_dim = int(config['MAD-GAN']['hidden_dim'])
        elif self.MODEL_TYPE in ['knn', 'gmm']:
            self.k = int(config['knn']['k'])
        elif self.MODEL_TYPE == 'RegGAN':
            self.g_lr = float(config['RegGAN']['g_lr'])
            self.d_lr = float(config['RegGAN']['d_lr'])
            self.E_lr = float(config['RegGAN']['E_lr'])
            self.z_dim = int(config['RegGAN']['z_dim'])
            self.filters = int(config['RegGAN']['filters'])
            self.kernel_size = int(config['RegGAN']['kernel_size'])
            self._lambda = float(config['RegGAN']['lambda'])
            self.n_critic = int(config['RegGAN']['n_critic'])
            self.gp_weight = float(config['RegGAN']['n_critic'])
            self.n_layers = int(config['RegGAN']['n_layers'])
            self.w_rec = float(config['RegGAN']['w_rec'])
            self.w_cx = float(config['RegGAN']['w_cx'])
            self.w_cz = float(config['RegGAN']['w_cz'])
            self.w_E = float(config['RegGAN']['w_E'])
            self.cz_layers = convert_string_to_list(config['RegGAN']['cz_layers'], int)
        elif self.MODEL_TYPE == 'lstm':
            self.n_layers = int(config['lstm']['n_layers'])
            self.hidden_dim = int(config['lstm']['hidden_dim'])
            self.lr = float(config['lstm']['lr'])
        elif self.MODEL_TYPE == 'TadGAN':
            self.g_lr = float(config['TadGAN']['g_lr'])
            self.d_lr = float(config['TadGAN']['d_lr'])
            self.z_dim = int(config['TadGAN']['z_dim'])
            self.filters = int(config['TadGAN']['filters'])
            self.G_n_layers = int(config['TadGAN']['G_n_layers'])
            self.hidden_dim = int(config['TadGAN']['G_n_layers'])
            self.kernel_size=int(config['TadGAN']['kernel_size'])
            self._lambda=float(config['TadGAN']['_lambda'])
            self.n_critic=int(config['TadGAN']['n_critic'])
            self.gp_weight=float(config['TadGAN']['gp_weight'])
            self.cz_layers = convert_string_to_list(config['TadGAN']['cz_layers'], int)


"""
Methods follow here that are used to create models using Config Object
"""
def define_beatgan(n_timesteps, n_features):
    g_lr = config.g_lr
    d_lr = config.d_lr

    filters = config.filters
    kernel_size = config.kernel_size
    z_dim = config.z_dim

    reg_w = config.reg_w

    return build_beatgan(
        n_timesteps,
        n_features,
        g_lr,
        d_lr,
        filters,
        kernel_size,
        z_dim,
        reg_w
    )

def define_madgan(n_timesteps, n_features):
    return build_madgan(
        n_timesteps=n_timesteps,
        n_features=n_features,
        g_lr=config.g_lr,
        d_lr=config.d_lr,
        z_dim=config.z_dim,
        n_layers=config.n_layers,
        hidden_dim=config.hidden_dim,
        use_1d_z=False,
    )


def define_reggan(n_timesteps, n_features):
    use_1d_z = True if config.DATASET == 'nasa' else False

    return build_reggan(
        n_timesteps=n_timesteps,
        n_features=n_features,
        g_lr=config.g_lr,
        d_lr=config.d_lr,
        E_lr=config.E_lr,
        z_dim=config.z_dim,
        filters=config.filters,
        kernel_size=config.kernel_size,
        _lambda=config._lambda,
        n_critic=config.n_critic,
        gp_weight=config.gp_weight,
        n_layers=config.n_layers,
        w_rec=config.w_rec,
        w_cx=config.w_cx,
        w_cz=config.w_cz,
        w_E=config.w_E,
        use_1d_z=use_1d_z,
        cz_layers=config.cz_layers if use_1d_z else None,
    )

def define_tadgan(n_timesteps, n_features):
    use_1d_z = True if config.DATASET == 'nasa' else False

    return build_tadgan(
        n_timesteps=n_timesteps,
        n_features=n_features,
        g_lr=config.g_lr,
        d_lr=config.d_lr,
        z_dim=config.z_dim,
        G_n_layers=config.G_n_layers,
        hidden_dim=config.hidden_dim,
        filters=config.filters,
        kernel_size=config.kernel_size,
        _lambda=config._lambda,
        n_critic=config.n_critic,
        gp_weight=config.gp_weight,
        cz_layers=config.cz_layers,
        use_1d_z=use_1d_z
    )


def define_knn():
    return KNN(config.k)


def define_gmm():
    return GMM(config.k)


def define_lstm(n_timesteps, n_features):
    return LSTMForecaster(
        num_classes=n_features,
        num_layers=config.n_layers,
        input_size=n_features,
        hidden_size=config.hidden_dim,
        seq_length=n_timesteps,
        lr=config.lr
    )

def save_model(model, trial_nr):
    if DATASET == 'smd':
        if MODEL_TYPE in ['BeatGAN', 'GAN', 'MAD_GAN']:
            torch.save(model.G.state_dict, 'saved_models/smd/1_1/{}/D/{}'.format(MODEL_TYPE, trial_nr))
            torch.save(model.D.state_dict, 'saved_models/smd/1_1/{}/G/{}'.format(MODEL_TYPE, trial_nr))


def get_model(model_type, n_timesteps, n_features):
    if model_type == 'BeatGAN':
        return define_beatgan(n_timesteps, n_features)
    elif model_type == 'RegGAN':
        return define_reggan(n_timesteps, n_features)
    elif model_type == 'knn':
        return define_knn()
    elif model_type == 'gmm':
        return define_gmm()
    elif model_type == 'lstm':
        return define_lstm(n_timesteps, n_features)
    elif model_type == 'TadGAN':
        return define_tadgan(n_timesteps, n_features)
    elif model_type == 'MAD-GAN':
        return define_madgan(n_timesteps, n_features)

def dummy_func(epoch):
    pass


def eval_model(pipeline):
    """
    Method that evlauates performance of models on given dataset
    args:
        pipeline: anomaly detection pipeline object
    returns:
        f1, precision, recall

    """
    if MODEL_TYPE in ['TadGAN']:
        with torch.backends.cudnn.flags(enabled=False):
            _ = pipeline.as_model.train(
                n_epochs=n_epochs,
                batch_size=batch_size,
                X_train=pipeline.X_train,
                is_param_tuning=True,
                tune_func=dummy_func
            )
        pipeline.as_model.set_models_to_eval()

    elif MODEL_TYPE in ['knn', 'gmm']:
        pipeline.as_model.fit(pipeline.train)
    elif MODEL_TYPE in ['lstm']:
        pipeline.as_model.fit(
            X_train=pipeline.X_train,
            n_epochs=n_epochs,
            batch_size=batch_size
        )
        pipeline.as_model.set_models_to_eval()
    else:
        _ = pipeline.as_model.train(
            n_epochs=n_epochs,
            batch_size=batch_size,
            X_train=pipeline.X_train,
            is_param_tuning=True,
            tune_func=dummy_func
        )

        pipeline.as_model.set_models_to_eval()

    f1_score, precision_score, recall_score, TP, TN, FP, FN = pipeline.predict_and_evaluate()
    print('f1: {} precision: {} recall: {} TP: {} TN {} FP {} FN {}'.format(f1_score, precision_score, recall_score, TP, TN, FP, FN))
    return f1_score, precision_score, recall_score

"""
Here comes code that reads configuaration file name from terminal, then reads that config file,
and develops tests using its contents.
"""

config_file_name = sys.argv[1]
config = Config(config_file_name)
MODEL_TYPE = config.MODEL_TYPE
DATASET = config.DATASET
n_steps = config.n_steps
n_epochs = config.n_epochs
batch_size = config.batch_size

if DATASET == 'nasa':
    f1_scores_smap = []
    precision_scores_smap = []
    recall_scores_smap = []

    f1_scores_msl = []
    precision_scores_msl = []
    recall_scores_msl = []
else:
    f1_scores = []
    precision_scores = []
    recall_scores = []

import sys

uncompleted = []
if DATASET == 'smd':
    for group_index in [1, 2, 3]:
        if group_index == 1:
            group_len = 8
        else:
            group_len = 9
        for index in range(1, group_len + 1):
            if (group_index, index) not in [(1,1), (1, 6), (1,7), (2,3), (3,2), (3,9)]:
                 continue
            try:
                print('-------performance on smd with group index {} and index {}'.format(group_index, index))
                args = {'group_index' : group_index, 'index' : index}
                pipeline = Pipeline(
                    None, DATASET, args, n_steps, batch_size, contamination_level=config.contamination_level
                )
                model = get_model(MODEL_TYPE, n_timesteps=n_steps, n_features=pipeline.X_train.shape[-1])
                pipeline.as_model = model
                f1, precision, recall = eval_model(pipeline)
                f1_scores.append(f1)
                precision_scores.append(precision)
                recall_scores.append(recall)
            except:
                uncompleted.append((group_index, index))
elif DATASET == 'nasa':
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'P', 'S', 'T']
    for letter in letters:
        for digit in range(1, 17):
            try:
                # the validation sets
                if digit == 1 and letter in ['A', 'F', 'P', 'T']:
                    continue
                file_name = '{}-{}'.format(letter, digit)
                args = {'file_name' : file_name}
                model = get_model(MODEL_TYPE, n_timesteps=n_steps, n_features=1)
                pipeline = Pipeline(model, DATASET, args, n_steps, batch_size, contamination_level=config.contamination_level)
            except:
                print(sys.exc_info()[0])
                continue
            print('-------performance on nasa with filename {}'.format(file_name))
            f1, precision, recall = eval_model(pipeline)
            if DATASET == 'nasa' and is_smap(file_name):
                f1_scores_smap.append(f1)
                precision_scores_smap.append(precision)
                recall_scores_smap.append(recall)
            elif DATASET == 'nasa' and not is_smap(file_name):
                f1_scores_msl.append(f1)
                precision_scores_msl.append(precision)
                recall_scores_msl.append(recall)
            else:
                f1_scores.append(f1)
                precision_scores.append(precision)
                recall_scores.append(recall)

print('############################################')
if DATASET == 'nasa':
    print('\n\n')
    print('f1 scores smap', f1_scores_smap)
    print('f1 scores msl', f1_scores_msl)
    print('mean smap', np.array(f1_scores_smap).mean())
    print('median smap', np.median(np.array(f1_scores_smap)))
    print('mean msl', np.array(f1_scores_msl).mean())
    print('median msl', np.median(np.array(f1_scores_msl)))
    print('\n\n')
    print('Precision scores smap', precision_scores_smap)
    print('Precision scores msl', precision_scores_msl)
    print('mean smap', np.array(precision_scores_smap).mean())
    print('median smap', np.median(np.array(precision_scores_smap)))
    print('mean msl', np.array(precision_scores_msl).mean())
    print('median msl', np.median(np.array(precision_scores_msl)))
    print('\n\n')
    print('Recall scores smap', recall_scores_smap)
    print('Recall scores msl', recall_scores_msl)
    print('mean smap', np.array(recall_scores_smap).mean())
    print('median smap', np.median(np.array(recall_scores_smap)))
    print('mean msl', np.array(recall_scores_msl).mean())
    print('median msl', np.median(np.array(recall_scores_msl)))
else:
    print('\n\n')
    print('f1 scores', f1_scores)
    print('mean', np.array(f1_scores).mean())
    print('median', np.median(np.array(f1_scores)))
    print('\n\n')
    print('Precision scores', precision_scores)
    print('mean', np.array(precision_scores).mean())
    print('median', np.median(np.array(precision_scores)))
    print('\n\n')
    print('Recall scores', recall_scores)
    print('mean', np.array(recall_scores).mean())
    print('median', np.median(np.array(recall_scores)))
    print('uncompleted', uncompleted)
    print('############################################')
