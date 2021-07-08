import optuna
from utils.pipeline import Pipeline
import sys
import torch
import numpy as np
from utils.optuna_utils import ModelFactory, ScoreTracker, save_hyperparameters, Config
from optuna.trial import TrialState

"""
Script that alllows one to perform optuna hyperparameter search for models on given datasets.

To perform hyperparameter searches on new models / datasets one can regard the code of this file as
an example.
"""


def objective(trial):
    """
    Objective to optimize per trial
    args:
        optuna trial
    returns:
        f1_score
    """

    def eval_anomalies(epoch, pipeline, score_tracker, on_last_set=False):
        """
        Function used at each epoch of methods to track of well they are doing thus far.
        args:
            pipeline: pipeline object
            score_tracker: object that tracks f1 scores
            on_last_set: boolean value that decides whether on last set
        returns:
            f1 score of method over evaluation set.
        """
        if epoch == n_epochs-1:
            return

        pipeline.as_model.set_models_to_eval()
        f1_score,_,_,_,_,_,_ = pipeline.predict_and_evaluate()
        pipeline.as_model.set_models_to_train()

        score_tracker.add_score(epoch, f1_score)
        if on_last_set:
            trial.report(
                score_tracker.get_avg(epoch),
                epoch
            )

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    def dummy_eval(epoch):
        pass

    def train_model(pipeline, batch_size, eval_func):
        """
        trains a given method for trial configurations
        """
        if MODEL_TYPE in ['TadGAN']:
            with torch.backends.cudnn.flags(enabled=False):
                _ = pipeline.as_model.train(
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    X_train=pipeline.X_train,
                    save_intermediate_models=False,
                    save_interval=0,
                    is_param_tuning=True,
                    tune_func=eval_func
                )
        elif MODEL_TYPE in ['knn', 'gmm']:
            pipeline.as_model.fit(pipeline.train)
        elif MODEL_TYPE in ['lstm']:
            pipeline.as_model.fit(
                X_train=pipeline.X_train,
                n_epochs=n_epochs,
                batch_size=batch_size
            )
        else:
            _ = pipeline.as_model.train(
                n_epochs=n_epochs,
                batch_size=batch_size,
                X_train=pipeline.X_train,
                save_intermediate_models=False,
                save_interval=0,
                is_param_tuning=True,
                tune_func=eval_func
            )

    def f1_score_smd():
        """
        function used to calculate f1 score produced by a model over smd dataset
        """
        n_steps_ceil = 40
        n_steps = trial.suggest_int("n_steps", 5, n_steps_ceil)
        batch_size = trial.suggest_int('batch_size', 20, 64)

        # definine model parameters for trial
        model_factory = ModelFactory(trial, n_timesteps=n_steps, n_features=38, model_type=MODEL_TYPE, dataset=DATASET)

        # create score tracker
        score_tracker = ScoreTracker()

        # average f1 scores over a few different datasets
        i_list = [1, 1, 3]
        j_list = [2, 4, 6]
        f1_values = []
        for index in range(len(i_list)):
            i, j = i_list[index], j_list[index]
            # make pipeline and model
            args = {'group_index': i, 'index': j}
            pipeline = Pipeline(None, DATASET, args, n_steps, batch_size)
            pipeline.as_model = model_factory.get_model(pipeline.X_train.shape[-1])

            on_last_dataset = True if index == len(i_list) - 1 else False
            train_model(pipeline, batch_size,
                        eval_func=lambda epoch: eval_anomalies(epoch, pipeline, score_tracker, on_last_dataset))
            f1_score_l, _, _, _, _, _, _ = pipeline.predict_and_evaluate()
            f1_values.append(f1_score_l)
        return np.array(f1_values).mean()

    def f1_score_nasa():
        """
        function used to calculate f1 score produced by a model over nasa dataset
        """
        # runs through the first dataset from some channels
        n_steps_ceil = 200
        n_steps = trial.suggest_int("n_steps", 5, n_steps_ceil)
        batch_size = trial.suggest_int('batch_size', 20, 100)

        # definine model parameters for trial
        model_factory = ModelFactory(trial, n_timesteps=n_steps, n_features=1, model_type=MODEL_TYPE, dataset=DATASET)

        # create score tracker
        score_tracker = ScoreTracker()

        letters = ['A', 'F', 'P', 'T']
        f1_values = []
        for i in range(len(letters)):
            letter = letters[i]
            file_name = '{}-{}'.format(letter, 1)
            args = {'file_name': file_name}
            model = model_factory.get_model(n_features=1)
            pipeline = Pipeline(model, DATASET, args, n_steps, batch_size)

            on_last_dataset = True if i == len(letters) - 1 else False
            train_model(pipeline, batch_size, eval_func=lambda epoch: eval_anomalies(epoch, pipeline, score_tracker, on_last_dataset))
            f1_score_l, _, _, _, _, _, _ = pipeline.predict_and_evaluate()
            f1_values.append(f1_score_l)
        return np.array(f1_values).mean()

    try:
        if DATASET == 'smd':
            f1_score = f1_score_smd()
        elif DATASET == 'nasa':
            f1_score = f1_score_nasa()
        else:
            raise NotImplementedError

    except RuntimeError as e:
        error_string = e.__str__().lower()
        legal_error_substrings = ['cuda', 'kernel size', 'cpuallocator', 'cudnn']
        for substring in legal_error_substrings:
            print(substring, error_string)
            if substring in error_string:
                raise optuna.exceptions.TrialPruned()
        raise e

    trial.report(f1_score, n_epochs-1)
    try:
        if f1_score > study.best_trial.value:
            save_hyperparameters(trial, f1_score, DATASET, MODEL_TYPE)

    except:
        save_hyperparameters(trial, f1_score, DATASET, MODEL_TYPE)
    return f1_score


"""
Here configuations are retrieved and the hyperparameter search is started
"""
config_file = sys.argv[1]
config = Config(config_file)
MODEL_TYPE = config.MODEL_TYPE
DATASET = config.DATASET


n_trials = config.n_trials
n_epochs = config.n_epochs

study = optuna.create_study(direction='maximize')
study.optimize(
    objective,
    n_trials=n_trials,
)


pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
