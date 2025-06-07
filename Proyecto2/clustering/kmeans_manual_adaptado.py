import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os

# === CONFIGURACIONES ===
PATH = {
    "features_csv": "../reduccion/data_reduccion/features_umap.csv",
    "movies_csv": "../data_MovieLens/movies.csv",
    "k_file": "./data_clustering/mejor_k.txt",
    "output_csv": "./data_clustering/embeddings_with_kmcustom.csv",
    "fig_clusters": "./fig_clustering/kmeans_custom.png"
}
os.makedirs("./data_clustering", exist_ok=True)
os.makedirs("./fig_clustering", exist_ok=True)

# =========================================
# 1) CARGAR DATOS
# =========================================

df = pd.read_csv(PATH["features_csv"])
X_umap = df.drop(columns=["movieId"]).values
movie_ids = df["movieId"].values
print(f"🔹 Datos cargados: {X_umap.shape[0]} muestras, {X_umap.shape[1]} dimensiones.")

# =========================================
# 2) LEER VALOR DE K
# =========================================
with open(PATH["k_file"], "r") as f:
    K_OPTIMO = int(f.read().strip())
print(f"🔹 Usando k = {K_OPTIMO} leído desde 'mejor_k.txt'.")

# =========================================
# 3) IMPLEMENTACIÓN DE K-MEANS CUSTOM
# =========================================

class KMeansCustom:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters   = n_clusters
        self.max_iter     = max_iter
        self.tol          = tol
        self.random_state = random_state
        self.centroids    = None
        self.labels       = None

    def fit(self, X):
        n_samples, n_features = X.shape
        if self.random_state is not None:
            np.random.seed(self.random_state)
        init_idxs = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[init_idxs].copy()

        for it in range(self.max_iter):
            dist_matrix = cdist(X, self.centroids, metric='euclidean')
            self.labels = np.argmin(dist_matrix, axis=1)

            new_centroids = np.zeros_like(self.centroids)
            for j in range(self.n_clusters):
                cluster_points = X[self.labels == j]
                if cluster_points.shape[0] > 0:
                    new_centroids[j] = cluster_points.mean(axis=0)
                else:
                    rand_idx = np.random.choice(n_samples)
                    new_centroids[j] = X[rand_idx]

            shifts = np.linalg.norm(new_centroids - self.centroids, axis=1)
            if np.max(shifts) < self.tol:
                self.centroids = new_centroids
                print(f"✅ KMeansCustom convergió en iteración {it+1}.")
                break
            self.centroids = new_centroids
        else:
            print(f"⚠️ KMeansCustom no convergió en {self.max_iter} iteraciones.")

    def predict(self, X_new):
        dist_matrix = cdist(X_new, self.centroids, metric='euclidean')
        return np.argmin(dist_matrix, axis=1)

# =========================================
# 4) EJECUTAR K-MEANS
# =========================================

kmeans = KMeansCustom(n_clusters=K_OPTIMO, max_iter=200, tol=1e-4, random_state=42)
kmeans.fit(X_umap)
labels_km = kmeans.labels

# =========================================
# 5) RESULTADOS Y VISUALIZACIÓN
# =========================================

# Guardar resultados
df_result = pd.DataFrame(X_umap, columns=["umap1", "umap2"])
df_result["movieId"] = movie_ids
df_result["cluster_km_custom"] = labels_km
df_result.to_csv(PATH["output_csv"], index=False)
print(f"💾 Resultados guardados en '{PATH['output_csv']}'")

# Visualizar
plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels_km, cmap='tab10', s=3, alpha=0.7)
plt.title(f"K-Means Custom (k = {K_OPTIMO})")
plt.tight_layout()
plt.savefig(PATH["fig_clusters"])
plt.close()
print(f"📊 Figura guardada en '{PATH['fig_clusters']}'")

# Mostrar distribución
print("\nDistribución de puntos por cluster:")
print(pd.Series(labels_km).value_counts().sort_index())

# Mostrar ejemplos de títulos
try:
    movies_df = pd.read_csv(PATH["movies_csv"], index_col="movieId")
    print("\nEjemplos de títulos por cluster:")
    for c in range(K_OPTIMO):
        ids_cluster = df_result[df_result["cluster_km_custom"] == c]["movieId"].tolist()
        muestra = ids_cluster[:10]
        print(f"\nCluster {c}:")
        for mid in muestra:
            title = movies_df.loc[mid, 'title'] if mid in movies_df.index else "(no disponible)"
            print(f"  • {mid}: {title}")
except Exception as e:
    print(f"⚠️ No se pudo mostrar títulos por cluster: {e}")
