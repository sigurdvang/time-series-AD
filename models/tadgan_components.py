import torch.nn as nn
import torch
import numpy as np
from models.model_utils import from_conv1d_format, to_conv1d_format

"""
This file contains the various components used by TadGAN
"""

class CnnDis(nn.Module):

    def __init__(self, n_features, filters, kernel_size, stride, padding, embed_size, sample_input_dim):
        super(CnnDis, self).__init__()
        self.main = self.build_encoder(n_features, filters, kernel_size, stride, padding, embed_size, sample_input_dim)

    @staticmethod
    def build_encoder(n_features, filters, kernel_size, stride, padding,embed_size, sample_input_dim) -> nn.Sequential:
        """
        Function that builds a nn.Sequential model, based on the data it will be working with, so that its convolutional
        operations creates a latent mapping as such (batch_size, n_features, seq_len) -> (batch_size, z_dim, 1)
        """
        layers = [
            nn.Conv1d(n_features, filters, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        sample_input = torch.randn(sample_input_dim)
        finished_building = False
        filter_exp = 1
        while not finished_building:
            temp_model = nn.Sequential(*layers)
            y = temp_model(sample_input)
            y_length = y.shape[2]

            # the case where one adds conv, norm, and activation layers
            if kernel_size < y_length:
                temp_layers = [
                    nn.Conv1d(filters ** filter_exp, filters ** (filter_exp + 1), kernel_size, stride, padding,
                              bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
                filter_exp += 1

            # the case where one adds the output layer
            else:
                out_kernel_size = y_length
                out_stride = 1
                out_padding = 0
                temp_layers = [
                    nn.Conv1d(filters ** filter_exp, embed_size, out_kernel_size, out_stride, out_padding, bias=False)
                ]
                finished_building = True

            layers.extend(temp_layers)

        return nn.Sequential(*layers)
        
    
    def forward(self, x):
        seq_len = x.shape[1]
        encoding = self.main(to_conv1d_format(x))
        return encoding

    
class FFDis(nn.Module):
    
    def __init__(self, n_features, seq_len, hidden_layers, use_1d_z):
        super(FFDis, self).__init__()
        self.use_1d_z = use_1d_z
        self.n_features = n_features
        self.seq_len = seq_len
        layers = []
        for i in range(len(hidden_layers)):
            units = hidden_layers[i]
            if i == 0:
                input_coefficient = 1 if self.use_1d_z else seq_len
                layers += [nn.Linear(n_features*input_coefficient, units)]
            else:
                prev_units = hidden_layers[i-1]
                layers += [nn.Linear(prev_units, units)]
            layers += [nn.ReLU()]
        self.features = nn.Sequential(*layers)
        if self.use_1d_z:
            layers = [nn.Linear(units, 1)]
        else:
            layers = [nn.Linear(units, seq_len)]
        
        self.classifier = nn.Sequential(*layers)
    
    def get_features(self, x):
        return self.features(x)
    
    def forward(self, x):
        batch_size = x.shape[0]
        if self.use_1d_z:
            return self.classifier(self.features(x))
        else:
            x = x.reshape(batch_size, self.seq_len*self.n_features)
            critic_scores = self.classifier(self.features(x))
            return critic_scores.reshape(batch_size, self.seq_len, 1)
    
class RnnDecoder(nn.Module):

    def __init__(self, embedding_dim, seq_len, n_layers, hidden_dim, n_features, dropout, use_1d_z):
        super(RnnDecoder, self).__init__()
        self.use_1d_z = use_1d_z
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_features = n_features
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_dim = hidden_dim
        self.num_directions = 2
        if n_layers > 1:
            self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True) 
        else:
            self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True) 
        self.decode_layer =  nn.Sequential(nn.Linear(hidden_dim * self.num_directions, n_features), nn.Tanh())

    def forward(self, x):
                
        batch_size = x.shape[0]
        
        if self.use_1d_z:
            x = x.repeat_interleave(self.seq_len, dim=1).view(batch_size, self.seq_len, -1)
        
        h_0 = torch.zeros(self.n_layers * self.num_directions, batch_size, self.hidden_dim).to(self.device)
        c_0 = torch.zeros(self.n_layers * self.num_directions, batch_size, self.hidden_dim).to(self.device)
        
        recurrent_features, (_,_) = self.lstm_layer(x, (h_0, c_0))
            
        decoding = self.decode_layer(recurrent_features.contiguous().view(batch_size*self.seq_len, self.hidden_dim * self.num_directions))
        decoding = decoding.view(batch_size, self.seq_len, self.n_features) 
        return decoding


class RnnEncoder(nn.Module):

    def __init__(self, in_dim, n_layers, hidden_dim, embedding_dim, dropout, is_discriminator=False, is_critic=False, use_1d_z=False):
        super(RnnEncoder, self).__init__()
        self.in_dim = in_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Encoder running on {}'.format(self.device))
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_directions = 2
        self.use_1d_z = use_1d_z
        self.is_discriminator = is_discriminator
        self.is_critic = is_critic
        if n_layers >  1:
            self.lstm_layer =  nn.LSTM(in_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        else:
            self.lstm_layer =  nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        
        if is_discriminator:
            self.encode_layer = nn.Sequential(nn.Linear(hidden_dim * self.num_directions, embedding_dim), nn.Sigmoid())
        elif is_critic:
            self.encode_layer = nn.Sequential(nn.Linear(hidden_dim * self.num_directions, embedding_dim))
        else:
            self.encode_layer = nn.Sequential(nn.Linear(hidden_dim * self.num_directions, embedding_dim), nn.Tanh())

    def is_regular_encoder(self):
        return not (self.is_critic or self.is_discriminator)

    def forward(self, x):
        self.batch_size, self.seq_len = x.shape[:2]
        h_0 = torch.zeros(self.n_layers * self.num_directions, self.batch_size, self.hidden_dim).to(self.device)
        c_0 = torch.zeros(self.n_layers * self.num_directions, self.batch_size, self.hidden_dim).to(self.device)
        recurrent_features, (h_end, _) = self.lstm_layer(x, (h_0, c_0))

        if self.use_1d_z and self.is_regular_encoder():
            if self.n_layers > 1:
                # selects only last layer
                h_end = h_end.view(self.n_layers, self.num_directions, self.batch_size, self.hidden_dim)[-1]
            h_end = h_end.contiguous().view(self.batch_size, self.hidden_dim * self.num_directions)
            encoding = self.encode_layer(h_end)
            return encoding.reshape((self.batch_size, self.embedding_dim))
        else:
            encoding = self.encode_layer(recurrent_features.contiguous().view(self.batch_size * self.seq_len, self.hidden_dim * self.num_directions))
            return encoding.reshape((self.batch_size, self.seq_len, self.embedding_dim))


class Autoencoder(nn.Module):
    
    def __init__(self, n_features, seq_len, embedding_dim, hidden_dim, n_layers, dropout, use_1d_z):
        super(Autoencoder, self).__init__()
        self.encoder = RnnEncoder(
            in_dim=n_features,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            use_1d_z=use_1d_z,
        )
        self.decoder = RnnDecoder(
            embedding_dim=embedding_dim,
            seq_len=seq_len,
            n_layers=n_layers,
            hidden_dim=hidden_dim, 
            n_features=n_features, 
            dropout=dropout,
            use_1d_z=use_1d_z,
        )
        self.use_1d_z = use_1d_z

    def forward(self, x):
        encoding = self.encoder(x)
        return self.decoder(
            encoding
        )