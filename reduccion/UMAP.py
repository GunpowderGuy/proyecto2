# UMAP.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import os

# === PATHS CENTRADOS ===
PATH = {
    "features_csv"      : "../extraccion/data_extraccion/visual_features.csv",
    "output_csv"        : "../reduccion/data_reduccion/features_umap.csv",
    "output_npy"        : "../reduccion/data_reduccion/features_umap.npy",
    "output_ids_npy"    : "../reduccion/data_reduccion/movie_ids_umap.npy",
    "fig_scatter"       : "./fig_reduccion/umap_2d.png"
}

def cargar_datos():
    print("=== 1. Cargar features visuales ===")
    df = pd.read_csv(PATH["features_csv"])
    X = df.drop(columns=["movieId"]).values
    movie_ids = df["movieId"].values
    return X, movie_ids

def aplicar_umap(X, n_neighbors=30, min_dist=0.1, n_components=2):
    print("=== 2. Normalizar y aplicar UMAP ===")
    X_scaled = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(n_components=n_components,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    return X_umap, X_scaled

def evaluar_reduccion(X_scaled, X_umap, n_clusters=10):
    print("=== 3. Evaluar métricas ===")
    trust = trustworthiness(X_scaled, X_umap, n_neighbors=15)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_umap)
    sil_score = silhouette_score(X_umap, labels, metric='euclidean')
    print(f"🔍 Trustworthiness: {trust:.4f}")
    print(f"🔍 Silhouette Score (k={n_clusters}): {sil_score:.4f}")

def guardar_resultados(X_umap, movie_ids):
    print("=== 4. Guardar resultados ===")
    os.makedirs(os.path.dirname(PATH["output_csv"]), exist_ok=True)
    df_umap = pd.DataFrame(X_umap, columns=['x', 'y'])
    df_umap["movieId"] = movie_ids
    df_umap.to_csv(PATH["output_csv"], index=False)
    np.save(PATH["output_npy"], X_umap)
    np.save(PATH["output_ids_npy"], movie_ids)

def visualizar_resultados(X_umap):
    print("=== 5. Visualización 2D ===")
    plt.figure(figsize=(8, 6))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], s=2, alpha=0.6)
    plt.title("UMAP: Visualización 2D")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PATH["fig_scatter"])
    plt.show()

if __name__ == "__main__":
    X, movie_ids = cargar_datos()
    X_umap, X_scaled = aplicar_umap(X, n_neighbors=30, min_dist=0.1)
    evaluar_reduccion(X_scaled, X_umap, n_clusters=10)
    guardar_resultados(X_umap, movie_ids)
    visualizar_resultados(X_umap)
