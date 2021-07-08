import torch.nn as nn
import torch
from models.model_utils import from_conv1d_format, to_conv1d_format
import numpy as np


class Encoder(nn.Module):
    """
    A simple CNN encoder, used for encoding sequences.
    """

    def __init__(self, n_features, filters, kernel_size, stride, padding, embed_size, sample_input_dim):
        super(Encoder, self).__init__()
        self.main = self.build_encoder(n_features, filters, kernel_size, stride, padding,embed_size, sample_input_dim)

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
                    nn.BatchNorm1d(filters ** (filter_exp + 1)),
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

    def get_layers(self):
        """
        Used by CNN discriminator so as to be able to create feature matching
        and classification sets of layers
        """

        def get_children(model: torch.nn.Module):
            layers = []
            children = list(model.children())
            if len(children) == 0:
                return model
            else:
                for child in children:
                    try:
                        layers.extend(get_children(child))
                    except TypeError:
                        layers.append(get_children(child))
            return layers
                        
        return get_children(self.main)

    def forward(self, x):
        encoding = self.main(to_conv1d_format(x))
        return encoding


class Decoder(nn.Module):
    """
    A simple CNN decoder, used for decoding sequences.
    """

    def __init__(self, n_features, filters, kernel_size, stride, padding, embed_size, out_length):
        super(Decoder, self).__init__()
        self.main = self.build_decoder(n_features, filters, kernel_size, stride, padding, embed_size, out_length)

    @staticmethod
    def build_decoder(n_features, filters, kernel_size, stride, padding, embed_size, out_length) -> nn.Sequential:
        """
        Function that builds a nn.Sequential model, based on the data it will be working with, so that its convolutional
        operations creates a latent mapping as such (batch_size, z_dim, 1) -> (batch_size, n_features, seq_len)
        """

        def recursively_add_layers(in_layers=[], in_filter_exp=1):
            """
            The need for this recursive function is because of the following. When iterativly building a decoder one
            has to start with the largest number of filters, and then work down from there. The problem is that one
            does not now what that number of filters will be, before one has found out how many layers one needs.
            Therefore, there is a need to redefine all layers at each iteration, rather than just append layers, as
            is done in the build_encoder method.

            Many of the parent function varuables are used as  "global" variables

            args:
                in_layer (list):
                    the list of layers one has recursively built thus far
                in_filter_exp (int):
                    the filter mult constant at this steo

            returns:
                list:
                    list of nn layers

            in_layers
            """

            # when one should add input layers
            if in_filter_exp == filter_exp:
                first_layers = [
                    nn.ConvTranspose1d(embed_size, filters ** in_filter_exp, kernel_size, stride, padding, bias=False),
                    nn.BatchNorm1d(filters ** in_filter_exp),
                    nn.ReLU(True)
                ]

                in_layers = first_layers + in_layers

            # when one should add hidden layers
            else:
                hidden_layers = [
                    nn.ConvTranspose1d(
                        filters ** (in_filter_exp + 1), filters ** in_filter_exp, kernel_size, stride, padding, bias=False
                    ),
                    nn.BatchNorm1d(filters ** in_filter_exp),
                    nn.ReLU(True),
                ]
                in_layers = hidden_layers + in_layers
                in_filter_exp += 1
                in_layers = recursively_add_layers(in_layers, in_filter_exp)

            return in_layers

        def get_y_length_after_computation():
            temp_layers = recursively_add_layers()
            temp_model = nn.Sequential(*temp_layers)
            y = temp_model(sample_input)
            ts_len = y.shape[2]
            len_diff = out_length - ts_len
            return ts_len, len_diff, temp_layers

        filter_exp = 0
        finished_building = False
        sample_input = torch.randn((50, embed_size, 1))
        layers = []

        while not finished_building:

            # get dimension of output data if layers are expanded
            filter_exp += 1
            y_length, _, _ = get_y_length_after_computation()

            # revert latest addition, and add output layer instead
            if y_length > out_length:
                filter_exp -= 1
                y_length, y_diff, layers = get_y_length_after_computation()

                out_stride = 1
                out_padding = 0
                # why add 1 to y_diff for kernel size? to get correct output size. Don't quite know why as of now
                output_layers = [
                    nn.ConvTranspose1d(filters, n_features, y_diff+1, out_stride, out_padding, bias=False),
                    nn.Tanh(),
                ]

                layers = layers + output_layers
                finished_building = True

        return nn.Sequential(*layers)
    
    def forward(self, x):
        return from_conv1d_format(
                self.main(x)
        )


class CnnAE(nn.Module):
    """
    A simple CNN Autoencoder, used for encoding sequences, and then decoding the encodings. Used for 
    anomaly detection as the Generator of various GAN models.
    """

    def __init__(self, n_features, filters, kernel_size, z_dim, sample_input_dim):
        super(CnnAE, self).__init__()
        self.encoder = Encoder(
            n_features=n_features,
            filters=filters,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
            embed_size=z_dim,
            sample_input_dim=sample_input_dim,
        )
        self.decoder = Decoder(
            n_features=n_features,
            filters=filters,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
            embed_size=z_dim,
            out_length=sample_input_dim[-1],
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        print('CNN_AE running on {}'.format(device))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded