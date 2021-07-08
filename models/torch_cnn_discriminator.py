import torch
import torch.nn as nn
from models.torch_cnn_ae import Encoder
from models.model_utils import to_conv1d_format, from_conv1d_format


class Dis(nn.Module):
    """
    A pytorch discriminator constructed using a CNN encoder.
    """

    def __init__(self, n_features, n_filters, kernel_size, sample_input_dim):
        super(Dis, self).__init__()

        # embed size = 1 because that is the truth label
        self.model = Encoder(
            n_features=n_features,
            filters=n_filters,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
            embed_size=1,
            sample_input_dim=sample_input_dim,
        )
        layers = self.model.get_layers()
        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        self.model.to(device)
        self.features.to(device)
        self.classifier.to(device)
        print('CNN Disc running on {}'.format(device))

    def get_features(self, x):
        x = to_conv1d_format(x)
        return self.features(x)

    def forward(self, x):
        x = to_conv1d_format(x)
        features = self.features(x)
        classifications = self.classifier(features)
        # sigmoid activation as we require truth labels between 0 and 1
        return torch.sigmoid(torch.squeeze(classifications, dim=2)) #, features

