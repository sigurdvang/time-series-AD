from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import numpy as np

class KNN:
    """
    Wrapper class to make sklearn KNN class fit seamlessly with the AD pipeline
    """
    def __init__(self, k):
        self.k = k
        self.nbrs = NearestNeighbors(n_neighbors=self.k)

    def fit(self, X):
        self.nbrs.fit(X)

    def point_wise_anomaly_score(self, X_windows, values, window_size):
        distances, _ = self.nbrs.kneighbors(values)
        return distances.mean(axis=1)


class GMM:
    """
    Wrapper class to make sklearn GM class fit seamlessly with the AD pipeline
    """
    def __init__(self, k):
        self.k = k
        self.model = GaussianMixture(n_components=k)

    def fit(self, X):
        self.model.fit(X)

    def point_wise_anomaly_score(self, X_windows, values, window_size):
        """
        Returns the highest probability for each point
        """
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min())

        scores = self.model.predict_proba(values)
        predictions = self.model.predict(values)
        max_scores = []
        for score in scores:
            max_scores.append(np.max(score))
        return normalize(-np.array(max_scores))
