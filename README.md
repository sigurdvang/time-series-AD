# Generative Adversarial Networks for Anomaly Detection in Multivariate Time-Series Data

This repo contains the code related to my master thesis of the same title. It was written in collaboration with Telenor, with the goal of evaluating the
utility of Generative Adversarial Networks (GANs) at performing anomaly detection (AD) in the telecommunicatins domain. As the Telenor dataset is not public, the code used
to process it is retracted from this repo, however, the model implementaions, as well as code used for experiments on benchmark datasets, still remain.

The GAN models implemented in this repo are MAD-GAN, BeatGAN and TadGAN, as well as a novel GAN-based AD method I call RegGAN. An AD pipeline for loading data, identifying 
anomalous subsequences, and scoring them accoring to F1-score, Precision and Recall is also included.

This repo is made public for reproducibility purposes, and its contents are explained in turn below.

## Root Folder
In the root folder there are two files. optuna_run_script.py is the script used for performing hyperparameter search with the different models on relevant datasets,
whilst test_script.py contains the code used to test the performance of models according to F1 score, Precision and Recall on given datasets. 
Both optuna_run_script.py as well as test_script.py requires there to be passed config files as command line parameters to run, containing system and model parameters. 

This can be done in the following way: test_scripy_py configs/test_config.txt


## Docker
Contains the Dockerfile and requirements needed to run this project.

## Configs
Contains various example configs for optuna and test scripts.

## models
Most crucially, this folder contains the file gan.py, where all the aforementioned GAN models are implemented. The file build_models.py contains functions for building
these models and gives an overview of what parameters they require. 

The GAN methods take pytorch modules as arguments to comprise their various components (e.g. Generator, Discriminator etc.). Examples of how these components might look 
are given in the remainding files. 

## saved_models 
A folder to save models during runtime

## utils
Utilities.

-evaluation.py has the code for finding and evaluating anomalies

-optuna_utils.py has utility code used for performing the optuna hyperparameter search for the various models. 

-pipeline.py includes the code for the AD pipeline

The folder utils/data contains further data-related utilities, detailed further within that folder.
