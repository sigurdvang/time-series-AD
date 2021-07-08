import torch
from torch import nn
from torch import from_numpy
import numpy as np


class LSTMForecaster(nn.Module):
    """
    A lstm forecaster designed to work seamlessly with the AD pipeline
    """

    def __init__(self, num_classes, num_layers, input_size, hidden_size, seq_length, lr=1e-3):
        super(LSTMForecaster, self).__init__()
        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True).to(self.device)

        self.fc = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh()).to(self.device)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        if self.num_layers > 1:
            h_out = h_out[-1]
        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out.reshape(out.shape[0], 1, out.shape[1])

    def create_batch(self, X_train, y_train, batch_size, batch_index):
        if (batch_index + batch_size) < X_train.shape[0]:
            X = X_train[batch_index:batch_index + batch_size]
            y = y_train[batch_index:batch_index + batch_size]
        else:
            X = X_train[batch_index:]
            y = y_train[batch_index:]
        return self.transform_input(X), self.transform_input(y)

    def create_labels(self, X_train):
        y = []
        for i in range(len(X_train) - 1):
            next_X = X_train[i + 1]
            y.append(next_X[0])
        return np.array(y)

    def fit(self, X_train, n_epochs, batch_size=64):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        y_train = self.create_labels(X_train)
        X_train = X_train[:-1]
        losses = []
        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                print(epoch)
            epoch_losses = []
            for batch_index in range(0, X_train.shape[0], batch_size):
                X, y = self.create_batch(X_train, y_train, batch_size, batch_index)
                optimizer.zero_grad()
                y_pred = self(X)
                y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[-1])
                loss = criterion(y_pred, y)
                epoch_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            losses.append(np.array(epoch_losses).mean())
        
        self.set_models_to_eval()
        return losses

    def transform_input(self, X):
        if torch.is_tensor(X):
            return X.to(torch.float32).to(self.device)
        return from_numpy(X).to(torch.float32).to(self.device)

    def point_wise_anomaly_score(self, X_windows, values, window_size):
        if not torch.is_tensor(X_windows):
            X_windows = self.transform_input(X_windows)

        predicts = []
        with torch.no_grad():
            predicts_windows = self(X_windows)
            for i in range(len(predicts_windows)):
                x = predicts_windows[i]
                predicts.append(x.reshape(x.shape[1]).cpu().numpy())

        predicts = np.array(predicts)
        actual = values[window_size:]

        anomaly_scores = np.mean(np.sqrt((predicts - actual) ** 2), 1)
        return anomaly_scores

    def set_models_to_train(self):
        self.train()

    def set_models_to_eval(self):
        self.eval()


