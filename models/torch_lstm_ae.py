import torch.nn as nn
import torch

class Encoder(nn.Module):
    """
    Pytorch Encoder. Encodes sequences.
    """

    def __init__(self, in_dim, n_layers, hidden_dim, embedding_dim, dropout):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Encoder running on {}'.format(self.device))
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        if n_layers >  1:
            self.lstm_layer =  nn.LSTM(in_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        else:
            self.lstm_layer =  nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.encode_layer = nn.Sequential(nn.Linear(hidden_dim, embedding_dim), nn.Tanh())



    def forward(self, x):
        h_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)
        c_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)
        recurrent_features, (_,_) = self.lstm_layer(x, (h_0, c_0))
        print(recurrent_features.shape)
        encoding = self.encode_layer(recurrent_features.contiguous().view(self.batch_size * self.seq_len, self.hidden_dim))
        print(encoding.shape)
        return encoding.reshape((self.batch_size, self.embedding_dim))

class Decoder(nn.Module):
    """
    Pytorch Decoder. Decoded encoded sequences.
    """

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
        x = x.repeat(1, self.seq_len, 1)
        x = x.reshape((self.batch_size, self.seq_len, self.embedding_dim))

        h_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)
        c_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)
        
        recurrent_features, (_,_) = self.lstm_layer(x, (h_0, c_0))
        decoding = self.decode_layer(recurrent_features.contiguous().view(self.batch_size*self.seq_len, self.hidden_dim))
        decoding = decoding.view(self.batch_size, self.seq_len, self.n_features) 
        return decoding

class RAE(nn.Module):
    """
    Lstm Autoencoder. Encodes sequences before decoding them into sequences meant to resemble the 
    original sequences. Used for anomaly detection, as the Generator of various GAN models.
    """
    def __init__(self, n_features, embedding_dim, hidden_dim, n_layers, dropout):
        super(RAE, self).__init__()
        self.encoder = Encoder(
            in_dim=n_features,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_features=n_features,
            dropout=dropout
        )

    def propegate_shape(self, batch_size, seq_len):
        self.encoder.batch_size = batch_size
        self.encoder.seq_len = seq_len
        self.decoder.batch_size = batch_size
        self.decoder.seq_len = seq_len

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        self.propegate_shape(batch_size, seq_len)
        return self.decoder(
            self.encoder(x)
        )
