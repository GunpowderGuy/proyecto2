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
    K_OPTIMO = int(f.read().strip())
print(f"🔹 Usando k = {K_OPTIMO} leído desde 'mejor_k.txt'.")

# === 2. Leer embeddings reducidos ===
df = pd.read_csv(FEATURES_CSV)
X = df.drop(columns=["movieId"]).values

# === 3. KMeansCustom mejorado ===

class KMeansImproved:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]

        for it in range(self.max_iter):
            distances = cdist(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[self.labels == j].mean(axis=0) if np.any(self.labels == j) else X[np.random.randint(n_samples)]
                for j in range(self.n_clusters)
            ])

            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                print(f"✅ Convergió en {it+1} iteraciones.")
                break
            self.centroids = new_centroids
        else:
            print(f"⚠️ No convergió en {self.max_iter} iteraciones.")

    def predict(self, X):
        return np.argmin(cdist(X, self.centroids), axis=1)


# Entrenar
kmc = KMeansImproved(n_clusters=K_OPTIMO, max_iter=300, tol=1e-4, random_state=42)
kmc.fit(X)
labels_manual = kmc.labels

# === 4. KMeans de sklearn ===
sk_model = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10)
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

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

print("Comparación KMeans Manual Adaptado vs sklearn")
print("Adjusted Rand Index (ARI):", adjusted_rand_score(labels_manual, labels_sklearn))
print("Normalized Mutual Info:", normalized_mutual_info_score(labels_manual, labels_sklearn))
print("V-measure:", v_measure_score(labels_manual, labels_sklearn))
