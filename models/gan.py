import torch
import torch.nn as nn
from torch import from_numpy
import numpy as np
from tqdm import trange
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Gan:
    """
    Class representing a Time-series Generative Adversarial Network meant for use in anomaly detection.
    It is therefore by default intended to have a autoencoder-based generator, as it is standard.
    The class can be subclassed into more specific / advanced GAN arhitectures.

    The GANs are designed to be agnostic to component architectures, meaning that for each GAN it
    should not matter whether a Generator has, for instance, recurrent or convolutional layers. Thererfore, 
    the neural network models used to comprise the GANs various compoents are passed in the constructor
    in the components argument.
    
    args:

        data_shape: is the shape of the data (seq_len, n_features)
        components: is a dictionary containing the makup of the GAN. See "build_models" to see
                    what components the various GANs need
    """

    def __init__(self, data_shape, components, model_name='GAN'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_shape = data_shape
        self.components = components
        self.model_name = model_name
        self.build_gan()

    def create_generator(self):
        # retrieving relevant components
        self.G = self.components['G']['model'].to(self.device)
        self.g_rec_loss_func = self.components['G']['loss']['reconstruction']
        self.g_gan_loss_func = self.components['G']['loss']['gan']
        opt = self.components['G']['opt']['gan']
        lr = self.components['G']['lr']
        self.g_gan_opt = opt(self.G.parameters(), lr=lr)

    def create_discriminator(self):
        # retrieving relevant components
        self.D = self.components['D']['model'].to(self.device)
        self.d_loss_func = self.components['D']['loss']
        opt = self.components['D']['opt']
        lr = self.components['D']['lr']
        self.d_opt = opt(self.D.parameters(), lr=lr)
    
    def set_critic_AD_specific_params(self):
        self.Lambda = self.components['gan']['lambda']

    def set_wgan_gp_specific_params(self):
        self.n_critic = self.components['gan']['n_critic']
        self.gp_weight = self.components['gan']['gp_weight']

    def build_gan(self):
        # create G and D
        self.create_generator()
        self.create_discriminator()

        # set GAN specific params
        self.reg_w = self.components['gan']['regularization_weight']
        self.use_1d_z = self.components['gan']['use_1d_z']

    def train_D(self, X):
        self.create_labels_for_training(X.shape[0])

        # get discriminator scores for real and fake data
        D_real = self.D(X)
        X_rec = self.G(X)
        D_fake = self.D(X_rec)
                
        # calculate loss
        D_real_loss = self.d_loss_func(D_real, self.real_labels)
        D_fake_loss = self.d_loss_func(D_fake, self.fake_labels)
        D_loss = D_real_loss + D_fake_loss
        
        # record loss
        self.history['epoch_d_adv_error'].append(D_loss.item())

        # perform backprop
        self.d_opt.zero_grad()
        D_loss.backward()
        self.d_opt.step()

    def get_G_adversarial_loss(self, X):
        return self.g_gan_loss_func(self.D(self.G(X)), self.real_labels)

    def get_G_reconstruction_loss(self, X):
        return self.g_rec_loss_func(self.G(X), X)

    def get_critic_score(self, X_windows, window_size, D=None):
        
        if D is None:
            D = self.D
            
        critic_scores = []
        with torch.no_grad():
            critic_windows = D(X_windows).detach().cpu().numpy()
            for i in range(len(X_windows)):
                c = critic_windows[i]
                critic_scores.append(
                    c.reshape(c.shape[0])[-1]
                )

        return np.asarray(critic_scores)

    def get_rec_score(self, X_windows, values, window_size, rec_func=None):
        """
        Method that calculates anomaly score using given model and data
        
        args:
            x_windows: subsequences of data to reconstruct
            values: the actual values to compare reconstruction against
            window_size: size of subsequences
            rec_func: function to use for reconstruction
     
        """

        if not torch.is_tensor(X_windows):
            X_windows = self.transform_input(X_windows)

        recons = []
        with torch.no_grad():
            recons_windows = self.reconstruct_np(X_windows, rec_func)
            for i in range(len(X_windows)):
                x = recons_windows[i]
                recons.append(
                    x.reshape(1, x.shape[0], x.shape[1])[:, -1, :]
                )

        actual = values[window_size:]
        recons = np.concatenate(recons, axis=0)

        anomaly_scores = np.mean(np.sqrt((recons - actual) ** 2), 1)
        return anomaly_scores


    def train_G(self, X):
        # calculate loss
        G_gan_loss = self.get_G_adversarial_loss(X)
        G_rec_loss = self.get_G_reconstruction_loss(X)
        G_loss = G_rec_loss + (G_gan_loss * self.reg_w)

        # record loss
        self.history['epoch_g_adv_error'].append(G_gan_loss.item())
        self.history['epoch_rec_error'].append(G_rec_loss.item())

        # perform backprop
        self.g_gan_opt.zero_grad()
        G_loss.backward()
        self.g_gan_opt.step()

    def create_batch(self, X_train, batch_size, batch_index):
        if (batch_index + batch_size) < X_train.shape[0]:
            batch = X_train[batch_index:batch_index+batch_size]
        else:
            batch = X_train[batch_index:]
        return batch

    def create_labels_for_training(self, batch_size):
        if self.use_1d_z:
            self.real_labels = torch.ones(batch_size, 1).to(self.device)
            self.fake_labels = torch.zeros(batch_size, 1).to(self.device)
        else:
            self.real_labels = torch.ones(batch_size, self.data_shape[0], 1).to(self.device)
            self.fake_labels = torch.zeros(batch_size, self.data_shape[0], 1).to(self.device)

    def create_history_object(self):
        self.history = {
            'rec_error'         : [],
            'g_adv_error'       : [],
            'd_adv_error'       : [],
            'epoch_rec_error'   : [],
            'epoch_g_adv_error' : [],
            'epoch_d_adv_error' : [],
        }

    def update_history(self):
        # update by taking the mean of epoch losses
        self.history['rec_error'].append(np.array(self.history['epoch_rec_error']).mean())
        self.history['g_adv_error'].append(np.array(self.history['epoch_g_adv_error']).mean())
        self.history['d_adv_error'].append(np.array(self.history['epoch_d_adv_error']).mean())

        # reset epoch losses
        self.history['epoch_rec_error'] = []
        self.history['epoch_g_adv_error'] = []
        self.history['epoch_d_adv_error'] = []
    
    def set_models_to_train(self):
        self.G = self.G.train()
        self.D = self.D.train()

    def set_models_to_eval(self):
        self.G = self.G.eval()
        self.D = self.D.eval()

    def train(self, n_epochs, batch_size, X_train, save_intermediate_models=False, save_interval=1, is_param_tuning=False, tune_func=None):
        """
        Training function. Allows for saving intermediate models, as well as to perform 
        callbacks used for parameter tuning. 

        args:
            n_epochs: epochs to train the model
            batch_size: the batch size of data during training
            X_train: the training data (size, sequence_length, n_features)
            save_intermediate_models: boolean flag denoting whether to save intermediate models
            save_interval: the interval at which models are saved
            is_param_tuning: boolean flag denoting whether one is performing parameter tuning
            tune_func: the function use to tune parameters
        """
        
        self.seq_len = X_train.shape[1]
        
        # self.set_models_to_train()
        print('Training {} on {}'.format(self.model_name, self.device))
        
        if is_param_tuning:
            print('--training with param tuning')
        
        # creates history of losses that can be used to track training of GAN
        self.create_history_object()
        
        # Because tqdm doesn not work well in Linux screen consoles
        if is_param_tuning:
            _range = range
        else:
            _range = trange
        
        for epoch in _range(n_epochs):
            self.epoch = epoch
            for batch_index in range(0, X_train.shape[0], batch_size):
                X = self.create_batch(X_train, batch_size, batch_index)
                X = self.transform_input(X)
                # train discriminator
                self.train_D(X)

                # train autoencoder through GAN and reconstruction
                self.train_G(X)

            self.update_history()

            if save_intermediate_models and ((epoch % save_interval) == 0 or epoch == n_epochs - 1):
                self.save_models(epoch)
                
            if is_param_tuning:
                tune_func(epoch)
                if epoch % 10 == 0:
                    print('epoch: {}'.format(epoch))

        self.set_models_to_eval()
        return self.history

    def summary(self):
        print(self.G)
        print(self.D)

    def transform_input(self, X):
        """
        Transforms input to a form that the model can work with
        args:
            X: data
        returns:
            X: data set to the device of the model, and converted to torch.float32
        """
        if torch.is_tensor(X):
            return X.to(torch.float32).to(self.device)
        return from_numpy(X).to(torch.float32).to(self.device)

    def reconstruct(self, X, to_cpu=True, rec_func=None):
        """
        reconstruct data
        """
        if rec_func is None:
            rec_func = self.G
        if not torch.is_tensor(X):
            X = self.transform_input(X)
        if to_cpu:
            return rec_func(X).detach().cpu()
        return rec_func(X).detach()

    def reconstruct_np(self, X, rec_func=None):
        """
        reconstruct data as numpy
        """
        if rec_func is None:
            rec_func = self.G
        if not torch.is_tensor(X):
            X = self.transform_input(X)
        return rec_func(X).detach().cpu().numpy()

    def discriminate_samples(self, X):
        """
        returns discrimination scores for samples
        """
        X = self.transform_input(X)
        return self.D(X).detach().cpu().numpy()

    def save_models(self, epoch, path='saved_models/{}/{}'):
        path = path.format(self.model_name, epoch)
        torch.save(self.G.state_dict(), path)

    def load_model_parameters(self, path):
        self.G.load_state_dict(torch.load(path))

    def _gradient_penalty(self, real_data, generated_data, D=None):
        """
        For GAN subclasses that uses the gradient penalty. This method produces the gradient penalty
        used to enforce the 1 lipschitz constraint for the wasserstein loss.
        """
        
        if D is None:
            D = self.D
        
        batch_size = real_data.size()[0]

        # Calculate interpolation
        if len(real_data.shape) == 3:
            alpha = torch.rand(batch_size, 1, 1)
        else:
            alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data).to(self.device)

        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = torch.autograd.Variable(interpolated, requires_grad=True).to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = D(interpolated)
        
        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
                outputs=prob_interpolated, 
                inputs=interpolated,
                grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                create_graph=True, 
                retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.reshape(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
    
    def feature_match(self, x, y, D=None):
        """
        Uses the discriminator to perform feature mathcing
        """
        if D is None:
            D = self.D
            
        return self.g_gan_loss_func(D.get_features(x), D.get_features(y).detach())
    
    def point_wise_anomaly_score(self, X_windows, values, window_size):
        """
        Metod to get anomaly score from gans
        Implemented in subclasses
        """
        raise NotImplementedError
        
class BeatGAN(Gan):
    """
    BeatGAN was introduced by Zhou et al in  www.ijcai.org/Proceedings/2019/0616.pdf
    For original implmentation go to: https://github.com/Vniex/BeatGAN/

    The main idea seperating it from normal autoencodered generator GANs, is that it uses 
    feature matching for adversarial regulation, and that it uses CNNs for both G and D.
    """

    def __init__(self, data_shape, components, model_name='BeatGAN'):
        super().__init__(data_shape, components, model_name)
    
    
    def point_wise_anomaly_score(self, X_windows, values, window_size):
        return self.get_rec_score(X_windows, values, window_size)
    
    
    def get_G_adversarial_loss(self, X):
        return self.feature_match(self.G(X), X, D=self.D)


class MadGAN(Gan):
    """
    MAD-GAN was introduced by Li et al in the following paper: https://arxiv.org/pdf/1901.04997.pdf
    It is a regular recurrent GAN, that assigns anomaly scores to samples by combining reconstruction
    errors and discriminator scores. 
    """
    
    def __init__(self, data_shape, components, z_dim, model_name='MadGAN'):
        self.z_dim = z_dim
        super().__init__(data_shape, components, model_name)

    def build_gan(self):
        # create G and D
        self.create_generator()
        self.create_discriminator()

        # set GAN specific params
        self.set_critic_AD_specific_params()
        self.use_1d_z = self.components['gan']['use_1d_z']
        
    def update_batch_size(self, batch_size):
        self.create_labels_for_training(batch_size)
        self.G.batch_size = batch_size
        self.D.layers.batch_size = batch_size
        
    def train_D(self, X):
        self.update_batch_size(X.shape[0])

        # get discriminator scores for real and fake data
        self.D_real = self.D(X)
        self.Z = self.sample_Z(X.shape[0])
        self.X_rec = self.G(self.Z)
        self.D_fake = self.D(self.X_rec)
        
        # calculate loss
        D_real_loss = self.d_loss_func(self.D_real, self.real_labels)
        D_fake_loss = self.d_loss_func(self.D_fake, self.fake_labels)
        D_loss = D_real_loss + D_fake_loss

        # record loss
        self.history['epoch_d_adv_error'].append(D_loss.item())

        # perform backprop
        self.d_opt.zero_grad()
        D_loss.backward()
        self.d_opt.step()
    
    def train_G(self, X):
        # calculate loss
        G_loss = self.get_G_adversarial_loss(self.Z)

        # record loss
        self.history['epoch_g_adv_error'].append(G_loss.item())
        self.g_gan_opt.zero_grad()
        G_loss.backward()
        self.g_gan_opt.step()        
    
    def update_history(self):
        # update by taking the mean of epoch losses
        self.history['g_adv_error'].append(np.array(self.history['epoch_g_adv_error']).mean())
        self.history['d_adv_error'].append(np.array(self.history['epoch_d_adv_error']).mean())

        # reset epoch losses
        self.history['epoch_g_adv_error'] = []
        self.history['epoch_d_adv_error'] = []
    
    def discriminate(self, X):
        X = self.transform_input(X)
        self.update_batch_size(X.shape[0])
        return self.D(X).detach().cpu().numpy()
    
    def get_rec_score(self, X_windows, values, window_size):
        """Method that calculates anomaly score using given model and data
        :param values: 2D array of multivariate time series data, shape (N, k)
        :param save_forecasts_and_recons: if True, saves forecasts and
        reconstructions together with anomaly score for each feature
        """

        recons = []
        Z = self.invert_samples(X_windows).detach().cpu().numpy()
        #Z = gan.sample_Z(X_windows.shape[0]).detach().cpu().numpy()
        with torch.no_grad():
            recons_windows = self.reconstruct_np(Z)
            for i in range(len(X_windows)):
                x = recons_windows[i]
                recons.append(
                    x.reshape(1, x.shape[0], x.shape[1])[:, -1, :]
                )

        actual = values[window_size:]
        recons = np.concatenate(recons, axis=0)

        anomaly_scores = np.mean(np.sqrt((recons - actual) ** 2), 1)
        return anomaly_scores
    
    def point_wise_anomaly_score(self, X_windows, values, window_size):
        if not torch.is_tensor(X_windows):
            X = self.transform_input(X_windows)
        else:
            X = window_set
            
        rec_score = self.get_rec_score(X, values, X.shape[1])
        critic_score = self.get_critic_score(X, X.shape[1])
                
        return ((1 - self.Lambda) * rec_score) + (self.Lambda * critic_score)
    

    def anomaly_score(self, X, z_k): 
        X_hat = self.G(z_k)
        mse_loss = torch.nn.MSELoss()
        residual_loss = mse_loss(X, X_hat)
        discrimination_loss = mse_loss(self.D(X), self.D(X_hat))
        total_loss = (1-self.Lambda)*residual_loss + self.Lambda*discrimination_loss
        return total_loss

    def invert_samples(self, X, epochs=5000, visualize=False):
        """
        Method that inverts samples. When MAD-GAN reconstructs samples, it needs to map them into
        latent space, so that they can be passed through the Generator. This method attempts to do so
        iterativly.

        args:
            X: subsequences
            epochs: number of iterations to find correct latent mapping
            visualize: boolean flag that determines whether or not to visualize the process

        """
        self.set_models_to_train()
        
        if not torch.is_tensor(X):
            X = self.transform_input(X)
    
        n_samples = X.shape[0]
        self.update_batch_size(n_samples)
        z_k = self.sample_Z(n_samples, True)
        optim = torch.optim.Adam([z_k], lr=1e-04)
        mse_loss = torch.nn.MSELoss()
        losses = []
        if visualize:
            _range = trange
        else:
            _range = range
            
        for i in _range(epochs):
            loss = self.anomaly_score(X, z_k)
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if (i % 1000 == 0 and i != 0) and visualize and False:
                plt.plot(losses)

        if visualize and False:
            plt.show()

        self.set_models_to_eval()
        return z_k.detach()
    
    def create_labels_for_training(self, batch_size):
        self.real_labels = torch.ones(batch_size, self.seq_len, 1).to(self.device)
        self.fake_labels = torch.zeros(batch_size, self.seq_len, 1).to(self.device)

    def sample_Z(self, batch_size, requires_grad=False):
        if self.use_1d_z:
            z_shape = (batch_size, self.z_dim)
        else:
            z_shape = (batch_size, self.seq_len, self.z_dim)

        z = torch.randn(z_shape).to(torch.float32).to(self.device)
        if requires_grad:
            z.requires_grad_(True)
        return z


class MadWGAN_GP(MadGAN):

    """
    Expanding MadGAN to be a wasserstein GAN with gradient penalties
    """

    def __init__(self, data_shape, components, z_dim, model_name='MadWGAN_GP'):
        super().__init__(data_shape, components, z_dim, model_name)

    def create_labels_for_training(self, batch_size):
        self.real_labels = torch.ones(batch_size, self.seq_len, 1).to(self.device)
        self.fake_labels = -torch.ones(batch_size, self.seq_len, 1).to(self.device)

    def create_generator(self):
        # retrieving relevant components
        self.G = self.components['G']['model'].to(self.device)
        opt = self.components['G']['opt']['gan']
        lr = self.components['G']['lr']
        self.g_gan_opt = opt(self.G.parameters(), lr=lr)

    def create_discriminator(self):
        # retrieving relevant components
        self.D = self.components['D']['model'].to(self.device)
        opt = self.components['D']['opt']
        lr = self.components['D']['lr']
        self.d_opt = opt(self.D.parameters(), lr=lr)

    def build_gan(self):
        # create G and D
        self.create_generator()
        self.create_discriminator()

        # set GAN specific params
        self.set_madgan_specific_params()
        self.set_wgan_gp_specific_params()

    def train_D(self, X):
        for n_i in range(1, self.n_critic+1):
            self.update_batch_size(X.shape[0])

            # get discriminator scores for real and fake data
            D_real = self.D(X)
            Z = self.sample_Z(X.shape[0])
            X_rec = self.G(Z)
            D_fake = self.D(X_rec)

            #calculate gradient penalties
            gradient_penalty = self._gradient_penalty(X, X_rec)
            # calculate wasserstein loss
            D_loss = -torch.mean(D_real) + torch.mean(D_fake) + gradient_penalty
            # record loss
            self.history['epoch_d_adv_error'].append(D_loss.item())

            # perform backprop
            self.d_opt.zero_grad()
            D_loss.backward()
            self.d_opt.step()

    def train_G(self, X):
        # calculate loss
        Z = self.sample_Z(X.shape[0])
        G_loss = -torch.mean(self.D(self.G(Z)))

        # record loss
        self.history['epoch_g_adv_error'].append(G_loss.item())
        self.g_gan_opt.zero_grad()
        G_loss.backward()
        self.g_gan_opt.step()


class TadGAN(Gan):
    """
    TadGAN was introduced by Geiger et al in the following paper: https://arxiv.org/abs/2009.07769

    It is a wasserstein GAN with an autoencoder-based generator, as well as two critics, C_x and C_z.
    C_x tries to meassure the distance between the distributions of real and synthetic samples, whilst
    C_z tries to meassure the distance between the distributions of latent represntations and white noise.

    The original implementation of TadGAN for univariate data can be found here: 
    https://github.com/signals-dev/Orion. This implementation differs by being in pytorch and by supporting
    multivariate time-series as well.
    """

    def __init__(self, data_shape, components, z_dim, model_name='TadGan'):
        self.z_dim = z_dim
        self.w_rec, self.w_adv = 10, 1
        super().__init__(data_shape, components, model_name)

    def create_history_object(self):
        self.history = {
            'g_adv_error': [],
            'cx_adv_error': [],
            'cz_adv_error': [],
            'epoch_g_adv_error': [],
            'epoch_cx_adv_error': [],
            'epoch_cz_adv_error': [],
        }

    def update_history(self):
        # update by taking the mean of epoch losses
        self.history['g_adv_error'].append(np.array(self.history['epoch_g_adv_error']).mean())
        self.history['cx_adv_error'].append(np.array(self.history['epoch_cx_adv_error']).mean())
        self.history['cz_adv_error'].append(np.array(self.history['epoch_cz_adv_error']).mean())

        # reset epoch losses
        self.history['epoch_rec_error'] = []
        self.history['epoch_g_adv_error'] = []
        self.history['epoch_cx_adv_error'] = []
        self.history['epoch_cz_adv_error'] = []
    
    def build_gan(self):
        # create G and D
        self.create_generator()
        self.create_discriminator()

        # set GAN specific params
        self.use_1d_z = self.components['gan']['use_1d_z']
        self.set_wgan_gp_specific_params()
        self.set_critic_AD_specific_params()
    
    def create_discriminator(self):
        # retrieving relevant components
        self.C_x = self.components['C_x']['model'].to(self.device)
        opt = self.components['C_x']['opt']
        lr = self.components['C_x']['lr']
        self.c_x_opt = opt(self.C_x.parameters(), lr=lr)

        self.C_z = self.components['C_z']['model'].to(self.device)
        opt = self.components['C_z']['opt']
        lr = self.components['C_z']['lr']
        self.c_z_opt = opt(self.C_z.parameters(), lr=lr)
        
    def create_generator(self):
        # retrieving relevant components
        self.G = self.components['G']['model'].to(self.device)
        self.g_rec_loss_func = self.components['G']['loss']['reconstruction']
        opt = self.components['G']['opt']['gan']
        lr = self.components['G']['lr']
        self.g_gan_opt = opt(self.G.parameters(), lr=lr)

    def train_D(self, X):
        
        def train_cx():
            sampled_Z = self.sample_Z(X.shape[0])
            X_fake = self.G.decoder(sampled_Z)
            
            # train C_X
            D_fake = self.C_x(X_fake)
            D_real = self.C_x(X)

            # calculate gradient penalties
            gradient_penalty = self._gradient_penalty(X, X_fake, D=self.C_x)
            # calculate wasserstein loss
            D_loss = torch.mean(D_fake) - torch.mean(D_real) + gradient_penalty

            # record loss
            self.history['epoch_cx_adv_error'].append(D_loss.item())

            # perform backprop
            self.c_x_opt.zero_grad()
            D_loss.backward()
            self.c_x_opt.step()
            
        def train_cz():
            sampled_Z = self.sample_Z(X.shape[0])
            encoded_Z = self.G.encoder(X)
            
            # train C_z
            D_fake = self.C_z(sampled_Z)
            D_real = self.C_z(encoded_Z)

            # calculate gradient penalties
            gradient_penalty = self._gradient_penalty(encoded_Z, sampled_Z, D=self.C_z)
            # calculate wasserstein loss
            D_loss =  torch.mean(D_fake) - torch.mean(D_real) + gradient_penalty

            # record loss
            self.history['epoch_cz_adv_error'].append(D_loss.item())

            # perform backprop
            self.c_z_opt.zero_grad()
            D_loss.backward()
            self.c_z_opt.step()
        
        for n_i in range(self.n_critic):
            train_cx()
            train_cz()
            

    def train_G(self, X):
        X_rec = self.G(X)
        
        # calculate loss
        G_loss = 0
        G_loss += self.get_G_reconstruction_loss(X, X_rec) * self.w_rec
        G_loss += self.get_G_adversarial_loss(X) * self.w_adv

        # record loss
        self.history['epoch_g_adv_error'].append(G_loss.item())
        self.g_gan_opt.zero_grad()
        G_loss.backward()
        self.g_gan_opt.step()

    def get_G_adversarial_loss(self, X):
        Z = self.sample_Z(X.shape[0])
        X_fake = self.G.decoder(Z)
        Z_fake = self.G.encoder(X)
        
        def loss_cx():
            return torch.mean(self.C_x(X)) - torch.mean(self.C_x(X_fake))
        
        def loss_cz():
            return torch.mean(self.C_z(Z)) - torch.mean(self.C_z(Z_fake))
        
        return loss_cx() + loss_cz()

    def get_G_reconstruction_loss(self, X, X_rec):
        return self.g_rec_loss_func(X_rec, X)

    def sample_Z(self, batch_size, requires_grad=False):
        if self.use_1d_z:
            z_shape = (batch_size, self.z_dim)
        else:
            z_shape = (batch_size, self.seq_len, self.z_dim)

        z = torch.randn(z_shape).to(torch.float32).to(self.device)
        if requires_grad:
            z.requires_grad_(True)
        return z
    
    def summary(self):
        print(self.G)
        print(self.C_x)
        print(self.C_z)
        
    def set_models_to_eval(self):
        self.G = self.G.eval()
        self.C_x = self.C_x.eval()
        self.C_z = self.C_z.eval()
        
    def set_models_to_train(self):
        self.G = self.G.train()
        self.C_x = self.C_x.train()
        self.C_z = self.C_z.train()

    def point_wise_anomaly_score(self, X_windows, values, window_size):
        """
        Calculates point-wise anomaly scores of samples. Works similar to that of other GAN models,
        combining critic and reconstruction scores. The implementation of use of critic scores is taken
        from the original TadGAN implementation https://github.com/signals-dev/Orion
        """
        
        if not torch.is_tensor(X_windows):
            X = self.transform_input(X_windows)
        else:
            X = X_windows
        
        def _process_critic_score(critics, smooth_window):
            critics = np.asarray(critics)
            l_quantile = np.quantile(critics, 0.25)
            u_quantile = np.quantile(critics, 0.75)
            in_range = np.logical_and(critics >= l_quantile, critics <= u_quantile)
            critic_mean = np.mean(critics[in_range])
            critic_std = np.std(critics)

            z_scores = np.absolute((np.asarray(critics) - critic_mean) / critic_std) + 1
            z_scores = pd.Series(z_scores).rolling(
                smooth_window, center=True, min_periods=smooth_window // 2).mean().values

            return z_scores


        critic_score = _process_critic_score(
            self.get_critic_score(X, X.shape[1], D=self.C_x), smooth_window = 50
        )
            
        rec_score = self.get_rec_score(X, values, window_size)
                
        return ((1 - self.Lambda) * rec_score) + (self.Lambda * critic_score)
    

class RegGAN(Gan):
    """
    Novel GAN architecture, combining several ideas from the state of the art.
    It is a wasserstein autoencoder-based GAN, with two critcs, and an additional encoder, to make it more
    robust to contamination. It also uses feature matching for further regularization.
   
    It is named RegGAN as it combines many state of the art regularization 
    techniques
    """
    
    def __init__(self, data_shape, components, z_dim,  model_name='Novel_GAN'):
        self.z_dim = z_dim
        super().__init__(data_shape, components, model_name)    

    def update_history(self):
        # update by taking the mean of epoch losses
        self.history['g_adv_error'].append(np.array(self.history['epoch_g_adv_error']).mean())
        self.history['E_rec_error'].append(np.array(self.history['epoch_E_rec_error']).mean())
        self.history['cx_adv_error'].append(np.array(self.history['epoch_cx_adv_error']).mean())
        self.history['cz_adv_error'].append(np.array(self.history['epoch_cz_adv_error']).mean())

        # reset epoch losses
        self.history['epoch_g_adv_error'] = []
        self.history['epoch_E_rec_error'] = []
        self.history['epoch_cx_adv_error'] = []
        self.history['epoch_cz_adv_error'] = []
    
    def create_history_object(self):
        self.history = {
            'g_adv_error': [],
            'E_rec_error': [],
            'cx_adv_error': [],
            'cz_adv_error': [],
            'epoch_g_adv_error': [],
            'epoch_E_rec_error': [],
            'epoch_cx_adv_error': [],
            'epoch_cz_adv_error': [],
        }
    
    def set_novelgan_specific_params(self):
        self.w_rec = self.components['gan']['w_rec']
        self.w_cx = self.components['gan']['w_cx']
        self.w_cz = self.components['gan']['w_cz']
        self.w_E = self.components['gan']['w_E']

    def build_gan(self):
        # create G and D
        self.create_generator()
        self.create_discriminator()

        # set GAN specific params
        self.use_1d_z = self.components["gan"]['use_1d_z']
        self.set_wgan_gp_specific_params()
        self.set_novelgan_specific_params()
        self.set_critic_AD_specific_params()
    
    def create_discriminator(self):
        # retrieving relevant components
        self.C_x = self.components['C_x']['model'].to(self.device)
        opt = self.components['C_x']['opt']
        lr = self.components['C_x']['lr']
        self.c_x_opt = opt(self.C_x.parameters(), lr=lr)

        self.C_z = self.components['C_z']['model'].to(self.device)
        opt = self.components['C_z']['opt']
        lr = self.components['C_z']['lr']
        self.c_z_opt = opt(self.C_z.parameters(), lr=lr)
        
    def create_generator(self):
        # retrieving relevant components
        self.G = self.components['G']['model'].to(self.device)
        self.g_rec_loss_func = self.components['G']['loss']['reconstruction']
        self.g_gan_loss_func = nn.MSELoss()
        opt = self.components['G']['opt']['gan']
        lr = self.components['G']['lr']
        self.g_gan_opt = opt(self.G.parameters(), lr=lr)
        
        self.E = self.components['E']['model'].to(self.device)
        self.e_loss_func = self.components['E']['loss']
        opt = self.components['E']['opt']
        lr = self.components['E']['lr']
        self.E_opt = opt(self.E.parameters(), lr=lr)

    def train_D(self, X):

        def train_cx():
            X_rec = self.G(X)

            # train C_X
            D_fake = self.C_x(X_rec)
            D_real = self.C_x(X)

            # calculate gradient penalties
            gradient_penalty = self._gradient_penalty(X, X_rec, D=self.C_x)
            # calculate wasserstein loss
            D_loss = torch.mean(D_fake) - torch.mean(D_real) + gradient_penalty

            # record loss
            self.history['epoch_cx_adv_error'].append(D_loss.item())

            # perform backprop
            self.c_x_opt.zero_grad()
            D_loss.backward()
            self.c_x_opt.step()

        def train_cz():
            sampled_Z = self.sample_Z(X.shape[0])
            encoded_Z = self.G.encoder(X)

            # train C_z
            D_fake = self.C_z(sampled_Z)
            D_real = self.C_z(encoded_Z)

            # calculate gradient penalties
            gradient_penalty = self._gradient_penalty(encoded_Z, sampled_Z, D=self.C_z)
            # calculate wasserstein loss
            D_loss = torch.mean(D_fake) - torch.mean(D_real) + gradient_penalty

            # record loss
            self.history['epoch_cz_adv_error'].append(D_loss.item())

            # perform backprop
            self.c_z_opt.zero_grad()
            D_loss.backward()
            self.c_z_opt.step()

        for n_i in range(self.n_critic):
            train_cx()
            train_cz()
            

    def train_G(self, X):
        X_rec = self.G(X)
        
        # calculate loss
        E_loss = self.get_E_loss(Z=self.G.encoder(X), X_rec=X_rec)
        G_loss = E_loss * self.w_E
        G_loss += self.get_G_reconstruction_loss(X, X_rec) * self.w_rec
        loss_cx, loss_cz = self.get_G_adversarial_loss(X, X_rec)
        G_loss += loss_cx * self.w_cx + loss_cz * self.w_cx

        # record loss
        self.history['epoch_g_adv_error'].append(G_loss.item())
        self.history['epoch_E_rec_error'].append(E_loss.item())

        # backprop G
        self.g_gan_opt.zero_grad()
        G_loss.backward() 
        self.g_gan_opt.step()

        # backprop E
        E_loss = self.get_E_loss(Z=self.G.encoder(X), X_rec=self.G(X))
        self.E_opt.zero_grad()
        E_loss.backward()
        self.E_opt.step()

    def get_G_adversarial_loss(self, X, X_rec):
        Z = self.sample_Z(X.shape[0])
        Z_fake = self.G.encoder(X)
        
        def loss_cx():
            return self.feature_match(X, X_rec, D=self.C_x)
        
        def loss_cz():
            return self.feature_match(Z, Z_fake, D=self.C_z)
        
        return loss_cx(),  loss_cz()

    def get_G_reconstruction_loss(self, X, X_rec):
        return self.g_rec_loss_func(X_rec, X)

    def get_E_loss(self, Z, X_rec):
        aux_Z = self.E(X_rec)
        return self.e_loss_func(Z, aux_Z)

    def sample_Z(self, batch_size, requires_grad=False):
        if self.use_1d_z:
            z_shape = (batch_size, self.z_dim)
        else:
            z_shape = (batch_size, self.seq_len, self.z_dim)

        z = torch.randn(z_shape).to(torch.float32).to(self.device)
        if requires_grad:
            z.requires_grad_(True)
        return z

    def point_wise_anomaly_score(self, X_windows, values, window_size):
        """
        Similar to how TadGAN computes anomaly scores, however, instead of using reconstruction error
        of samples, it uses reconstruction error between latent encoding of the Generator and 
        the auxiliary encoder.
        """

        if not torch.is_tensor(X_windows):
            X = self.transform_input(X_windows)
        else:
            X = X_windows

        def _process_critic_score(critics, smooth_window):
            critics = np.asarray(critics)
            l_quantile = np.quantile(critics, 0.25)
            u_quantile = np.quantile(critics, 0.75)
            in_range = np.logical_and(critics >= l_quantile, critics <= u_quantile)
            critic_mean = np.mean(critics[in_range])
            critic_std = np.std(critics)

            z_scores = np.absolute((np.asarray(critics) - critic_mean) / critic_std) + 1
            z_scores = pd.Series(z_scores).rolling(
                smooth_window, center=True, min_periods=smooth_window // 2).mean().values

            return z_scores

        critic_score = _process_critic_score(
            self.get_critic_score(X, X.shape[1], D=self.C_x), smooth_window=50
        )
        
        G_encodings = self.G.encoder(X)
        E_encodings = self.E(self.G.decoder(G_encodings))
        G_encodings, E_encodings = G_encodings.detach().cpu().numpy(), E_encodings.detach().cpu().numpy()
        if not self.use_1d_z:
            unrolled_GE = []
            unrolled_EE = []
            for i in range(len(X)):
                z = G_encodings[i]
                ze = E_encodings[i]
                unrolled_GE.append(
                    z.reshape(1, z.shape[0], z.shape[1])[:, -1, :]
                )
                unrolled_EE.append(
                    ze.reshape(1, ze.shape[0], ze.shape[1])[:, -1, :]
                )

            unrolled_GE = np.concatenate(unrolled_GE, axis=0)
            unrolled_EE = np.concatenate(unrolled_EE, axis=0)

            rec_score = np.mean(np.sqrt((unrolled_EE - unrolled_GE) ** 2), 1)
        
        else:
            rec_score = np.mean(np.sqrt((E_encodings - G_encodings) ** 2), 1)

        return ((1 - self.Lambda) * rec_score) + (self.Lambda* critic_score)

        
    def summary(self):
        print(self.G)
        print(self.E)
        print(self.C_x)
        print(self.C_z)
        
    def set_models_to_train(self):
        self.G = self.G.train()
        self.E = self.E.train()
        self.C_x = self.C_x.train()
        self.C_z = self.C_z.train()
        
    def set_models_to_eval(self):
        self.G = self.G.eval()
        self.E = self.E.eval()
        self.C_x = self.C_x.eval()
        self.C_z = self.C_z.eval()
