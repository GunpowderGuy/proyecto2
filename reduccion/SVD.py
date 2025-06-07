# SVD.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
import os

# === PATHS CENTRADOS ===
PATH = {
    "features_csv"      : "../extraccion/data_extraccion/visual_features.csv",
    "output_csv"        : "../reduccion/data_reduccion/features_svd.csv",
    "output_npy"        : "../reduccion/data_reduccion/features_svd.npy",
    "output_ids_npy"    : "../reduccion/data_reduccion/movie_ids_svd.npy",
    "fig_2d"            : "./fig_reduccion/svd_2d_visual_check.png",
    "fig_hist"          : "./fig_reduccion/svd_distance_distribution.png"
}

def cargar_datos():
    print("=== 1. Cargar features visuales ===")
    df = pd.read_csv(PATH["features_csv"])
    X = df.drop(columns=["movieId"]).values
    movie_ids = df["movieId"].values
    return X, movie_ids

def aplicar_svd(X, n_componentes=2000):
    print(f"=== 2. Aplicar SVD con {n_componentes} componentes ===")
    X_scaled = StandardScaler().fit_transform(X)
    svd = TruncatedSVD(n_components=n_componentes, random_state=42)
    X_svd = svd.fit_transform(X_scaled)
    return X_svd, svd

def guardar_resultados(X_svd, movie_ids):
    print("=== 3. Guardar resultado ===")
    os.makedirs(os.path.dirname(PATH["output_csv"]), exist_ok=True)

    df_svd = pd.DataFrame(X_svd)
    df_svd["movieId"] = movie_ids
    df_svd.to_csv(PATH["output_csv"], index=False)

    np.save(PATH["output_npy"], X_svd)
    np.save(PATH["output_ids_npy"], movie_ids)

def visualizar_componentes(X_svd):
    print("=== 4. Visualizar primeras 2 componentes ===")
    plt.figure(figsize=(8, 6))
    plt.scatter(X_svd[:, 0], X_svd[:, 1], s=5, alpha=0.5, c='green')
    plt.title("Visualización SVD (primeras 2 componentes)")
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PATH["fig_2d"])
    plt.show()

def graficar_distancias(X_svd, sample_size=500):
    print(f"=== 5. Evaluar distribución de distancias entre {sample_size} muestras ===")
    np.random.seed(42)
    indices = np.random.choice(X_svd.shape[0], sample_size, replace=False)
    sample = X_svd[indices, :]
    distances = pdist(sample, metric='euclidean')

    plt.figure(figsize=(8, 5))
    sns.histplot(distances, bins=50, kde=True, color='darkgreen', alpha=0.7)
    plt.title("Distribución de distancias euclidianas\nentre 500 muestras (SVD features)")
    plt.xlabel("Distancia euclidiana")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(PATH["fig_hist"])
    plt.show()

if __name__ == "__main__":
    X, movie_ids = cargar_datos()
    X_svd, svd = aplicar_svd(X, n_componentes=2000)
    guardar_resultados(X_svd, movie_ids)
    print(f"Componentes retenidos: {svd.n_components}")
    print(f"Varianza explicada total: {svd.explained_variance_ratio_.sum():.4f}")
    visualizar_componentes(X_svd)
    graficar_distancias(X_svd)