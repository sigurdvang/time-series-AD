import configparser
from models.build_model import build_reggan, build_tadgan, build_beatgan
from models.sklearn_wrappers import KNN, GMM
from models.torch_lstm_forecaster import LSTMForecaster
import torch
import numpy as np


class Config:
    """
    Simple class that handles that retrieves configurations
    for the system to perform optuna hyperparameter search
    """

    def __init__(self, config_file_name):
        conf = configparser.ConfigParser()
        conf.read(config_file_name)

        print(conf.keys())

        self.MODEL_TYPE = conf['SYSTEM']['MODEL_TYPE']
        self.DATASET = conf['SYSTEM']['DATASET']
        self.n_trials = int(conf['SYSTEM']['n_trials'])
        self.n_epochs = int(conf['SYSTEM']['n_epochs'])


class ModelFactory:
    """
    Class that defines search space of hyperparameter search, and build models using configuaratios
    choosen for given trials
    """

    def __init__(self, trial, n_timesteps, n_features, model_type, dataset):
        self.trial = trial
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.model_type = model_type
        self.dataset = dataset

        if model_type == 'BeatGAN':
            self.define_beatgan_params()
        elif model_type == 'TadGAN':
            self.define_tadgan_params()
        elif model_type == 'RegGAN':
            self.define_reggan_params()
        elif model_type == 'knn':
            self.define_knn_params()
        elif model_type == 'gmm':
            self.define_gmm_params()
        elif model_type == 'lstm':
            self.define_lstm_params()

    def define_beatgan_params(self):
        self.g_lr = 1e-03
        self.d_lr = 1e-03

        self.filters = self.trial.suggest_int("filters", 4, 32)
        self.kernel_size = self.trial.suggest_int("kernel_size", 4, 32)
        if self.dataset == 'smd':
            self.z_dim = self.trial.suggest_int("z_dim", 1, int(self.n_timesteps * 0.9))
        else:
            self.z_dim = self.trial.suggest_int('z_dim', 1, int(self.n_timesteps * 0.9))

        self.reg_w = self.trial.suggest_categorical('reg_w', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    def define_tadgan_params(self):
        self.use_1d_z = True if self.dataset == 'nasa' else False

        if self.use_1d_z:
            self.z_dim = self.trial.suggest_int('z_dim', 1, int(self.n_timesteps * 0.9))
        else:
            self.z_dim = self.trial.suggest_int("z_dim", 1, int(self.n_features * 0.9))

        self.g_lr = 5e-05
        self.d_lr = 5e-05
        self.G_n_layers = self.trial.suggest_int("G_n_layers", 1, 2)
        self.hidden_dim = self.trial.suggest_int("hidden_dim", 50, 150)
        self.filters = self.trial.suggest_int("filters", 8, 32)
        self.kernel_size = self.trial.suggest_int("kernel_size", 12, 32)
        self._lambda = self.trial.suggest_float("_lambda", 0.1, 0.9)
        self.n_critic = self.trial.suggest_int("n_critic", 2, 5)
        self.gp_weight = self.trial.suggest_int("gp_weight", 1, 20)

        cz_n_layers = self.trial.suggest_int("cz_n_layers", 1, 2)
        self.cz_layers = []
        for i in range(cz_n_layers):
            self.cz_layers.append(
                self.trial.suggest_int("cz_layer_{}".format(i), 20, 100)
            )

    def define_reggan_params(self):
        self.use_1d_z = True if self.dataset == 'nasa' else False

        self.g_lr = 5e-05
        self.E_lr = 5e-05
        self.d_lr = 5e-05

        if self.use_1d_z:
            self.z_dim = self.trial.suggest_int("z_dim", 1, int(self.n_timesteps * 0.9))
        else:
            self.z_dim = self.trial.suggest_int('z_dim', 1, int(self.n_features * 0.9))

        self.filters = self.trial.suggest_int("filters", 8, 100)
        self.kernel_size = self.trial.suggest_int("kernel_size", 4, max(5, int(self.n_timesteps * 0.7)))
        self._lambda = self.trial.suggest_float("_lambda", 0.1, 0.9)
        self.n_critic = self.trial.suggest_int("n_critic", 2, 5)
        self.gp_weight = self.trial.suggest_int("gp_weight", 1, 20)
        self.n_layers = self.trial.suggest_int('n_layers', 1, 4)
        self.w_rec = self.trial.suggest_int('w_rec', 1, 10)
        self.w_cx = self.trial.suggest_int('w_cx', 1, 10)
        self.w_cz = self.trial.suggest_int('w_cz', 1, 10)
        self.w_E = self.trial.suggest_int('w_E', 1, 10)

        if self.use_1d_z:
            cz_n_layers = self.trial.suggest_int("cz_n_layers", 1, 7)
            self.cz_layers = []
            for i in range(cz_n_layers):
                self.cz_layers.append(
                    self.trial.suggest_int("cz_layer_{}_nodes".format(i), 20, 200)
                )
        else:
            self.cz_layers = None

    def define_knn_params(self):
        self.k = self.trial.suggest_int('k', 1, 200)

    def define_gmm_params(self):
        self.k = self.trial.suggest_int('k', 1, 200)

    def define_lstm_params(self):
        self.n_layers = self.trial.suggest_int('n_layers', 1, 7)
        self.hidden_dim = self.trial.suggest_int('hidden_dim', self.n_timesteps, 500)
        self.lr = self.trial.suggest_float('lr', 1e-5, 1e-2)

    def get_model(self, n_features=None):
        """
        n_features of class is of dataset family
        however, features can change from set to set
        """
        if n_features is None:
            n_features = self.n_features

        if self.model_type == 'BeatGAN':
            return build_beatgan(
                self.n_timesteps,
                n_features,
                self.g_lr,
                self.d_lr,
                self.filters,
                self.kernel_size,
                self.z_dim,
                self.reg_w
            )
        elif self.model_type == 'TadGAN':
            return build_tadgan(
                n_timesteps=self.n_timesteps,
                n_features=n_features,
                g_lr=self.g_lr,
                d_lr=self.d_lr,
                z_dim=self.z_dim,
                G_n_layers=self.G_n_layers,
                hidden_dim=self.hidden_dim,
                filters=self.filters,
                kernel_size=self.kernel_size,
                _lambda=self._lambda,
                n_critic=self.n_critic,
                gp_weight=self.gp_weight,
                cz_layers=self.cz_layers,
                use_1d_z=self.use_1d_z,
            )
        elif self.model_type == 'RegGAN':
            return build_reggan(
                self.n_timesteps,
                n_features,
                self.g_lr,
                self.E_lr,
                self.d_lr,
                self.z_dim,
                self.filters,
                self.kernel_size,
                self._lambda,
                self.n_critic,
                self.gp_weight,
                self.n_layers,
                self.w_rec,
                self.w_cx,
                self.w_cz,
                self.w_E,
                self.use_1d_z,
                cz_layers=self.cz_layers
            )
        elif self.model_type == 'knn':
            return KNN(self.k)
        elif self.model_type == 'gmm':
            return GMM(self.k)
        elif self.model_type == 'lstm':
            return LSTMForecaster(
                num_classes=n_features,
                num_layers=self.n_layers,
                input_size=n_features,
                hidden_size=self.hidden_dim,
                seq_length=self.n_timesteps,
                lr=self.lr
            )


class ScoreTracker:
    """
    Simple class that tracks F1 scores during training of models
    """

    def __init__(self):
        self.scores = {}

    def add_score(self, epoch, score):
        if epoch in self.scores.keys():
            self.scores[epoch].append(score)
        else:
            self.scores[epoch] = [score]

    def get_avg(self, epoch):
        return np.array(self.scores[epoch]).mean()


def save_hyperparameters(trial, f1, dataset, model_type):
    f = open('saved_models/{}/{}/hyperparams.txt'.format(dataset, model_type), 'a')
    append_string = 'trial {} on smd  \n'.format(trial.number)
    f.write(append_string)
    append_string = '    f1 score: {} \n'.format(f1)
    f.write(append_string)
    for k, v in trial.params.items():
        append_string = "        {}: {} \n".format(k, v)
        f.write(append_string)


def save_model(model, trial_nr, dataset, model_type):
    if dataset == 'smd':
        if model_type in ['BeatGAN', 'GAN', 'MAD_GAN']:
            torch.save(model.G.state_dict, 'saved_models/smd/1_1/{}/D/{}'.format(model_type, trial_nr))
            torch.save(model.D.state_dict, 'saved_models/smd/1_1/{}/G/{}'.format(model_type, trial_nr))
        elif model_type == 'TadGAN':
            torch.save(model.G.state_dict, 'saved_models/smd/1_1/{}/G/{}'.format(model_type, trial_nr))
            torch.save(model.C_z.state_dict, 'saved_models/smd/1_1/{}/C_z/{}'.format(model_type, trial_nr))
            torch.save(model.C_x.state_dict, 'saved_models/smd/1_1/{}/C_x/{}'.format(model_type, trial_nr))
