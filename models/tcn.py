import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

"""
The code of this file implements temporal convolutional networks used for generators and critics of
various GAN methods. The ideas of temporal convolutional networks as they are used here was introduced by 
Bai et al in the following paper: https://arxiv.org/abs/1803.01271

The code is based off their companion repo found here: https://github.com/locuslab/TCN

As well as this repo, who shows this method applied to gans: 
https://github.com/proceduralia/pytorch-GAN-timeseries
"""

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
 
        self.network = nn.Sequential(*layers)

    def  forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
          self.linear.weight.data.normal_(0, 0.01)
    
    
    def forward(self, x, channel_last=True):
        #If channel_last, the expected format is (batch_size, seq_len, features)
        y1 = self.tcn(x.transpose(1, 2) if channel_last else x)
        return self.linear(y1.transpose(1, 2))


class CausalConvEncoder(nn.Module):
  
    def __init__(self, input_size, embedding_dim, n_layers, n_channel, kernel_size, dropout=0, use_1d_z=False, seq_len=None):
        super().__init__()
        self.use_1d_z = use_1d_z
        #Assuming same number of channels layerwise
        num_channels = [n_channel] * n_layers

        if not use_1d_z:
            self.tcn = TCN(input_size, embedding_dim, num_channels, kernel_size, dropout)
        else:
            # if using 1d_z the downsampling is done in this model
            self.tcn = TCN(input_size, input_size, num_channels, kernel_size, dropout)
            self.down_sampler = nn.Linear(in_features=input_size*seq_len, out_features=embedding_dim)

    def forward(self, x, channel_last=True):
        encoding = self.tcn(x, channel_last)

        if self.use_1d_z:
            batch_size, seq_len, n_features = x.shape
            encoding = self.down_sampler(encoding.reshape(batch_size, seq_len * n_features))

        return torch.tanh(encoding)

class CausalConvDecoder(nn.Module):
 
    def __init__(self, embedding_dim, output_size, n_layers, n_channel, kernel_size, dropout=0, use_1d_z=False, seq_len=None):
        super().__init__()
        self.use_1d_z = use_1d_z
        self.seq_len = seq_len
        num_channels = [n_channel] * n_layers
        self.tcn = TCN(embedding_dim, output_size, num_channels, kernel_size, dropout)
        
    def forward(self, x, channel_last=True):
        batch_size = x.shape[0]
        if self.use_1d_z:
            x = x.repeat_interleave(self.seq_len, dim=1).view(batch_size, self.seq_len, -1)
        return torch.tanh(self.tcn(x, channel_last))


class CausalConvAutoencoder(nn.Module):
    
    def __init__(self, input_size, embedding_dim, n_layers, n_channel, kernel_size, dropout=0, use_1d_z=False, seq_len=False):
        super().__init__()
        self.use_1d_z = use_1d_z
        #Assuming same number of channels layerwise
        num_channels = [n_channel] * n_layers
        self.use_1d_z = use_1d_z
        self.encoder = CausalConvEncoder(input_size, embedding_dim, n_layers, n_channel, kernel_size, dropout, use_1d_z, seq_len)
        self.decoder = CausalConvDecoder(embedding_dim, input_size, n_layers, n_channel, kernel_size, dropout, use_1d_z, seq_len)
    
    
    def forward(self, X):
        encoding = self.encoder(X)
        return self.decoder(
            encoding
        )
    
class CausalConvDisc(nn.Module):
    
    def __init__(self, input_size, n_layers, n_channel, kernel_size, dropout=0):
        super().__init__()
        #Assuming same number of channels layerwise
        num_channels = [n_channel] * n_layers
        self.model = TCN(input_size, 1, num_channels, kernel_size, dropout=0)
        self.features = self.model.tcn
        self.classifier = self.model.linear
    
    def get_features(self, X):
        return self.features(X.transpose(1, 2))
    
    def forward(self, X):
        X = X.transpose(1, 2)
        features = self.features(X)
        return self.classifier(features.transpose(1, 2))