import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist

class RBFNN_Improved:
    def __init__(self, n_centers=100, sigma_mode='local', local_k=5, sigma_scale=1.0, ridge_alpha=1e-2, random_state=42):
        self.n_centers = n_centers
        self.sigma_mode = sigma_mode
        self.local_k = local_k
        self.sigma_scale = sigma_scale
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state

    def _compute_local_sigmas(self, centers):
        nbrs = NearestNeighbors(n_neighbors=min(self.local_k+1, len(centers)), algorithm='auto').fit(centers)
        dists, _ = nbrs.kneighbors(centers)
        local = dists[:, 1:]  
        sigmas = local.mean(axis=1)
        sigmas[sigmas == 0] = np.mean(sigmas[sigmas > 0])
        return sigmas

    def _compute_global_sigma(self, centers):
        d = pdist(centers)
        s = d.mean() * self.sigma_scale
        if s == 0:
            s = 1.0
        return s

    def _rbf_matrix(self, X, centers, sigmas):
        X2 = np.sum(X**2, axis=1)[:, None]   
        C2 = np.sum(centers**2, axis=1)[None, :]  
        XC = X.dot(centers.T)  
        dist_sq = X2 + C2 - 2*XC
        if np.isscalar(sigmas):
            H = np.exp(-dist_sq / (2.0 * (sigmas**2)))
        else:
            sig2 = (sigmas**2)[None, :]
            H = np.exp(-dist_sq / (2.0 * sig2))
        return H

    def fit(self, X, y):
        enc = OneHotEncoder(sparse_output=False)
        Y = enc.fit_transform(y.reshape(-1,1))
        self.encoder = enc

        kmeans = KMeans(n_clusters=self.n_centers, random_state=self.random_state, n_init=10).fit(X)
        self.centers = kmeans.cluster_centers_

        if self.sigma_mode == 'local':
            sigmas = self._compute_local_sigmas(self.centers)
            sigmas = sigmas * self.sigma_scale
            self.sigmas = sigmas
            sigma_info = "local (per-center)"
        else:
            s = self._compute_global_sigma(self.centers)
            if self.sigma_mode == 'scaled':
                s = s * self.sigma_scale
            self.sigmas = s
            sigma_info = f"global sigma={s:.4g}"

        Phi = self._rbf_matrix(X, self.centers, self.sigmas)

        ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        ridge.fit(Phi, Y)
        self.W = ridge.coef_.T
        self.ridge = ridge
    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def predict_proba(self, X):
        Phi = self._rbf_matrix(X, self.centers, self.sigmas)
        scores = Phi.dot(self.W)
        probs = self._softmax(scores)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))