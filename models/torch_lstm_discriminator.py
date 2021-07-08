import torch.nn as nn
import torch
from models.torch_lstm_ae import Encoder

class Dis(nn.Module):
    """
    Pytorch lstm discriminator
    Built using a pytorch encoder
    """
    
    def __init__(self, embedding_dim, n_layers, hidden_dim,  n_features, dropout):
        super(Dis, self).__init__()        
        self.encoder = Encoder(n_features, n_layers, hidden_dim, embedding_dim, dropout)
        
    def forward(self, x):
        return torch.sigmoid(torch.squeeze(self.encoder(x), dim=2))