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
# 3) IMPLEMENTACIÓN DE K-MEANS MEJORADO
# =========================================
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

# =========================================
# 4) EJECUTAR K-MEANS
# =========================================
kmeans = KMeansImproved(n_clusters=K_OPTIMO, max_iter=300, tol=1e-4, random_state=42)
kmeans.fit(X_umap)
labels_km = kmeans.labels

# =========================================
# 5) RESULTADOS Y VISUALIZACIÓN
# =========================================
df_result = pd.DataFrame(X_umap, columns=["umap1", "umap2"])
df_result["movieId"] = movie_ids
df_result["cluster_km_custom"] = labels_km
df_result.to_csv(PATH["output_csv"], index=False)
print(f"💾 Resultados guardados en '{PATH['output_csv']}'")

plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels_km, cmap='tab10', s=3, alpha=0.7)
plt.title(f"K-Means Mejorado (k = {K_OPTIMO})")
plt.tight_layout()
plt.savefig(PATH["fig_clusters"])
plt.close()
print(f"📊 Figura guardada en '{PATH['fig_clusters']}'")

print("\nDistribución de puntos por cluster:")
print(pd.Series(labels_km).value_counts().sort_index())

try:
    movies_df = pd.read_csv(PATH["movies_csv"], index_col="movieId")
    print("\nEjemplos de títulos por cluster:")
    for c in range(K_OPTIMO):
        ids_cluster = df_result[df_result["cluster_km_custom"] == c]["movieId"].tolist()[:10]
        print(f"\nCluster {c}:")
        for mid in ids_cluster:
            title = movies_df.loc[mid, 'title'] if mid in movies_df.index else "(no disponible)"
            print(f"  • {mid}: {title}")
except Exception as e:
    print(f"⚠️ No se pudo mostrar títulos por cluster: {e}")
