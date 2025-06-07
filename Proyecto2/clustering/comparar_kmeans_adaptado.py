import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import cdist
import os

# === RUTAS ===
FEATURES_CSV = "../reduccion/data_reduccion/features_umap.csv"
K_FILE = "./data_clustering/mejor_k.txt"
OUTPUT_PNG = "./fig_clustering/kmeans_comparacion_adaptado.png"

os.makedirs("./fig_clustering", exist_ok=True)

# === 1. Leer mejor k ===
with open(K_FILE, "r") as f:
    k_optimo = int(f.read().strip())
print(f"📌 k seleccionado: {k_optimo}")

# === 2. Leer embeddings reducidos ===
df = pd.read_csv(FEATURES_CSV)
X = df.drop(columns=["movieId"]).values

# === 3. KMeansCustom (adaptado) ===
class KMeansCustom:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters   = n_clusters
        self.max_iter     = max_iter
        self.tol          = tol
        self.random_state = random_state
        self.centroids    = None
        self.labels       = None

    def fit(self, X):
        n_samples, _ = X.shape
        if self.random_state is not None:
            np.random.seed(self.random_state)
        idxs = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idxs].copy()

        for i in range(self.max_iter):
            dists = cdist(X, self.centroids, metric='euclidean')
            self.labels = np.argmin(dists, axis=1)

            new_centroids = np.empty_like(self.centroids)
            for j in range(self.n_clusters):
                cluster_points = X[self.labels == j]
                if cluster_points.shape[0] > 0:
                    new_centroids[j] = cluster_points.mean(axis=0)
                else:
                    rand_idx = np.random.choice(n_samples)
                    new_centroids[j] = X[rand_idx]

            shift = np.linalg.norm(new_centroids - self.centroids, axis=1).max()
            if shift < self.tol:
                break
            self.centroids = new_centroids

# Entrenar
kmc = KMeansCustom(n_clusters=k_optimo, max_iter=200, tol=1e-4, random_state=42)
kmc.fit(X)
labels_manual = kmc.labels

# === 4. KMeans de sklearn ===
sk_model = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
labels_sklearn = sk_model.fit_predict(X)

# === 5. Calcular ARI ===
ari = adjusted_rand_score(labels_manual, labels_sklearn)
print(f"✅ ARI entre manual y sklearn: {ari:.4f}")

# === 6. Visualizar ===
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels_manual, s=3, cmap='tab10', alpha=0.7)
plt.title("KMeans Manual Adaptado")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_sklearn, s=3, cmap='tab10', alpha=0.7)
plt.title("KMeans (scikit-learn)")

plt.tight_layout()
plt.savefig(OUTPUT_PNG)
print(f"📊 Figura guardada en {OUTPUT_PNG}")
