"""
Util methods for the various models

They are all related to permuting data
"""

def to_conv1d_format(x):
    """
    (batch_size, seq_len, n_features) -> (batch_size, n_features, seq_len)
    """
    return x.permute(0, 2, 1)

def from_conv1d_format(x):
    """
    (batch_size, n_features, seq_len) -> (batch_size,, n_features, seq_len)
    """
    return x.permute(0, 2, 1)

def to_batch_first_format(x):
    return x.permute(1, 0, 2)

def from_batch_first_format(x):
    return x.permute(1, 0, 2)