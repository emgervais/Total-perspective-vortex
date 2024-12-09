import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        cov_matrix = np.cov(X_centered.T)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_var = eigenvalues.sum()
        explained_variance_ratio = eigenvalues / total_var
        cumsum_variances = np.cumsum(explained_variance_ratio)
        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            n_components = np.sum(cumsum_variances <= self.n_components) + 1
        else:
            n_components = X.shape[1] if self.n_components is None else int(self.n_components)
            
        self.components_ = eigenvectors.T[:n_components]
        self.explained_variance_ = eigenvalues[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:n_components]
        
        return self

    def transform(self, X, y=None):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)