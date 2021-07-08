import torch.nn as nn
import torch

"""
This file contains the various components used by MAD-GAN
"""

class Encoder(nn.Module):

    def __init__(self, in_dim, n_layers, hidden_dim, embedding_dim, dropout, is_discriminator=False, is_critic=False, use_1d_z=False):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Encoder running on {}'.format(self.device))
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.use_1d_z = use_1d_z
        if n_layers >  1:
            self.lstm_layer =  nn.LSTM(in_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        else:
            self.lstm_layer =  nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        
        if is_discriminator:
            self.encode_layer = nn.Sequential(nn.Linear(hidden_dim, embedding_dim), nn.Sigmoid())
        elif is_critic:
            self.encode_layer = nn.Sequential(nn.Linear(hidden_dim, embedding_dim))
        else:
            self.encode_layer = nn.Sequential(nn.Linear(hidden_dim, embedding_dim), nn.Tanh())

    def is_regular_encoder(self):
        return not (self.is_critic or self.is_discriminator)

    def forward(self, x):
        self.batch_size, self.seq_len = x.shape[:2]
        h_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)
        c_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)
        recurrent_features, (h_end,_) = self.lstm_layer(x, (h_0, c_0))

        if self.use_1d_z and self.is_regular_encoder():
            h_end = h_end.contiguous().view(self.batch_size, self.hidden_dim)
            encoding = self.encode_layer(h_end)
            return encoding.reshape((self.batch_size, self.embedding_dim))
        else:
            encoding = self.encode_layer(recurrent_features.contiguous().view(self.batch_size * self.seq_len, self.hidden_dim))
            return encoding.reshape((self.batch_size, self.seq_len, self.embedding_dim))


class Decoder(nn.Module):

    def __init__(self, embedding_dim, n_layers, hidden_dim, n_features, dropout):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_features = n_features
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_dim = hidden_dim
        if n_layers > 1:
            self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True) 
        else:
            self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True) 
        self.decode_layer =  nn.Sequential(nn.Linear(hidden_dim, n_features), nn.Tanh())


    def forward(self, x):
        self.batch_size, self.seq_len = x.shape[:2]


        h_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)
        c_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)
        
        recurrent_features, (_,_) = self.lstm_layer(x, (h_0, c_0))
        decoding = self.decode_layer(recurrent_features.contiguous().view(self.batch_size*self.seq_len, self.hidden_dim))
        decoding = decoding.view(self.batch_size, self.seq_len, self.n_features) 
        return decoding


class Dis(nn.Module):
    
    def __init__(self, n_layers, hidden_dim, n_features, dropout, is_wgan=False):
        super(Dis, self).__init__() 
        if is_wgan:
            self.layers = Encoder(n_features, n_layers, hidden_dim, 1, dropout, is_critic=True)
        else:
            self.layers = Encoder(n_features, n_layers, hidden_dim, 1, dropout, is_discriminator=True)
        
    def forward(self, x):
        return self.layers(x)
    
    
class Autoencoder(nn.Module):
    
    def __init__(self, n_features, embedding_dim, hidden_dim, n_layers, dropout):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(
            in_dim=n_features,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim, 
            n_features=n_features, 
            dropout=dropout
        )


    def forward(self, x):
        encoding = self.encoder(x)
        if self.use_1d_z:
            batch_size, seq_len = x.shape[:-1]
            encoding = encoding.repeat_interleave(seq_len, dim=1).view(batch_size, seq_len, -1)
            print('AE encoding', encoding.shape)
        return self.decoder(
            encoding
        )
    
if __name__ == '__main__':
    
    n_features = 30
    dis = Dis(2, 100, n_features, dropout=0.2)
    dis.to(dis.layers.device)
    dis.layers.seq_len = 50
    dis.layers.batch_size = 10
    
    x = torch.randn((dis.layers.batch_size, dis.layers.seq_len, n_features)).to(torch.float32).to(dis.layers.device)
    score = dis(x)
    print(score.shape)