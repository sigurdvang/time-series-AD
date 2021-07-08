from models.gan import TadGAN, RegGAN, BeatGAN, MadGAN
from models.tadgan_components import Autoencoder, CnnDis, FFDis
from models.torch_cnn_ae import CnnAE
from models.torch_cnn_discriminator import Dis
from models.tcn import *
import torch.nn.functional as F
from models.madgan_components import Decoder as MADGAN_Decoder
from models.madgan_components import Dis as MADGAN_Dis

"""
This file contains methods to make construction of the modethods used easier

Several of the GAN-based methods have parameters called use_1d_z. This is in reference to how the latent 
dimension of the models should be. If use_1d_z == True, then encoder mapping of the model will look something
like this

(size, seq_len, n_features) -> (size, z_dim)

otherwise, it will look like this.

(size, seq_len, n_features) -> (size, seq_len, z_dim)

I've found that the former is more common when working with univariate data, whilst the latter is more common
when working with multivariate data.
"""

def build_tadgan(
    n_timesteps, 
    n_features,
    g_lr,
    d_lr,
    z_dim,
    G_n_layers,
    hidden_dim,
    filters,
    kernel_size,
    _lambda,
    n_critic,
    gp_weight,
    cz_layers,
    use_1d_z,
):


    data_shape = (n_timesteps, n_features)
   
    G = Autoencoder(
        n_features=n_features, 
        embedding_dim=z_dim,
        seq_len=n_timesteps,
        hidden_dim=hidden_dim, 
        n_layers=G_n_layers, 
        dropout=0.25,
        use_1d_z=use_1d_z
    )

    c_x = CnnDis(
        n_features=n_features,
        filters=filters,
        kernel_size=kernel_size,
        sample_input_dim = (10, n_features, n_timesteps),
        stride=2, 
        padding=1, 
        embed_size=n_timesteps
    )
    c_z = FFDis(
        n_features=z_dim,
        seq_len = n_timesteps,
        hidden_layers = cz_layers,
        use_1d_z=use_1d_z,
    )


    components = {
        'G': {
            'model': G,
            'opt': {
                'gan': torch.optim.RMSprop,
            },
            'loss' : {
                'reconstruction': torch.nn.MSELoss(),
            },
            'lr': g_lr,
        },
        'C_x': {
            'model': c_x,
            'opt': torch.optim.RMSprop,
            'lr': d_lr,
        },
        'C_z': {
            'model': c_z,
            'opt': torch.optim.RMSprop,
            'lr': d_lr,
        },
        'gan': {
            'lambda': _lambda,
            'n_critic': n_critic,
            'gp_weight': gp_weight,
            'use_1d_z': use_1d_z,
            
        }

    }
    
    return TadGAN(data_shape=data_shape, components=components, z_dim=z_dim)


def build_reggan(
    n_timesteps,
    n_features,
    g_lr,
    E_lr,
    d_lr,
    z_dim,
    filters,
    kernel_size,
    _lambda,
    n_critic,
    gp_weight,
    n_layers,
    w_rec,
    w_cx,
    w_cz,
    w_E,
    use_1d_z,
    cz_layers=None
):
    data_shape = (n_timesteps, n_features)
    

    G = CausalConvAutoencoder(
        input_size=n_features, 
        embedding_dim=z_dim, 
        n_layers=n_layers, 
        n_channel=filters, 
        kernel_size=kernel_size, 
        dropout=0.2,
        use_1d_z=use_1d_z,
        seq_len=n_timesteps
    )

    E = CausalConvEncoder(
        input_size=n_features, 
        embedding_dim=z_dim, 
        n_layers=n_layers, 
        n_channel=filters, 
        kernel_size=kernel_size, 
        dropout=0.2,
        use_1d_z=use_1d_z,
        seq_len=n_timesteps,
    )

    c_x = CausalConvDisc(
        input_size=n_features, 
        n_layers=n_layers, 
        n_channel=filters, 
        kernel_size=kernel_size, 
        dropout=0.2,
    )
    
    
    if use_1d_z:
        c_z = FFDis(
            n_features=z_dim,
            seq_len = n_timesteps,
            hidden_layers = cz_layers,
            use_1d_z=True,
        )
    else:
        c_z = CausalConvDisc(
            input_size=z_dim, 
            n_layers=n_layers, 
            n_channel=filters, 
            kernel_size=kernel_size, 
            dropout=0.2,
        )


    components = {
        'G': {
            'model': G,
            'opt': {
                'gan': torch.optim.RMSprop,
            },
            'loss' : {
                'reconstruction': torch.nn.MSELoss(),
            },
            'lr': g_lr,
        },
        'E' : {
            'model': E,
            'opt': torch.optim.RMSprop,
            'loss': torch.nn.MSELoss(),
            'lr': E_lr,
        },
        'C_x': {
            'model': c_x,
            'opt': torch.optim.RMSprop,
            'lr': d_lr,
        },
        'C_z': {
            'model': c_z,
            'opt': torch.optim.RMSprop,
            'lr': d_lr,
        },
        'gan': {
            'lambda': _lambda,
            'n_critic': n_critic,
            'gp_weight': gp_weight,
            'w_rec': w_rec,
            'w_cx': w_cx,
            'w_cz': w_cz,
            'w_E': w_E,
            'use_1d_z': use_1d_z,
        }

    }

    return RegGAN(data_shape, components, z_dim)


def build_beatgan(
    n_timesteps,
    n_features,
    g_lr,
    d_lr,
    filters,
    kernel_size,
    z_dim,
    reg_w,
):
    data_shape = (n_timesteps, n_features)
    sample_input_dim = (10, n_features, n_timesteps)

    G = CnnAE(
        n_features=n_features,
        filters=filters,
        kernel_size=kernel_size,
        z_dim=z_dim,
        sample_input_dim=sample_input_dim,
    )

    D = Dis(
        n_features=n_features,
        n_filters=filters,
        kernel_size=kernel_size,
        sample_input_dim=sample_input_dim,
    )

    components = {
        'G': {
            'model': G,
            'loss': {
                'reconstruction': torch.nn.MSELoss(),
                'gan': torch.nn.MSELoss(),
            },
            'opt': {
                'gan': torch.optim.Adam,
            },
            'lr': g_lr,
        },
        'D': {
            'model': D,
            'loss': F.binary_cross_entropy,
            'opt': torch.optim.RMSprop,
            'lr': d_lr,
        },
        'gan': {
            'regularization_weight': reg_w,
            'use_1d_z': True,
        }
    }

    return BeatGAN(data_shape=data_shape, components=components)


def build_madgan(
    n_timesteps,
    n_features,
    g_lr,
    d_lr,
    z_dim,
    n_layers,
    hidden_dim,
    use_1d_z,
):

    data_shape = (n_timesteps, n_features)


    G = MADGAN_Decoder(
        embedding_dim = z_dim,
        n_layers = n_layers,
        hidden_dim = hidden_dim,
        n_features = n_features,
        dropout=0.25
    )

    D = MADGAN_Dis(
        n_features=n_features,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        dropout=0.25
    )

    components = {
        'G' : {
            'model': G,
            'loss' : {
                'reconstruction': torch.nn.MSELoss(),
                'gan': torch.nn.MSELoss(),
            },
            'opt'  : {
                'gan' :torch.optim.Adam,
            },
            'lr' : g_lr,
        },
        'D' : {
            'model': D,
            'loss' : F.binary_cross_entropy,
            'opt'  : torch.optim.RMSprop,
            'lr'   : d_lr,
        },
        'gan' : {
            'lambda' : 0.1,
            'use_1d_z' : use_1d_z,
        }
    }

    return MadGAN(data_shape=data_shape, z_dim=z_dim, components=components)
    