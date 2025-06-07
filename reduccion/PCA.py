# PCA.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import os

# === PATHS CENTRADOS ===
PATH = {
    "visual_features_csv": "../extraccion/data_extraccion/visual_features.csv",
    "output_csv"         : "../reduccion/data_reduccion/features_pca.csv",
    "output_npy"         : "../reduccion/data_reduccion/features_pca.npy",
    "output_ids_npy"     : "../reduccion/data_reduccion/movie_ids_pca.npy",
    "fig_pca_2d"         : "./fig_reduccion/pca_2d_visual_check.png",
    "fig_hist"           : "./fig_reduccion/distribucion_distancias_pca.png"
}

def cargar_datos():
    print("=== 1. Cargar visual_features.csv ===")
    df = pd.read_csv(PATH["visual_features_csv"])
    movie_ids = df["movieId"].values
    X = df.drop(columns=["movieId"]).values
    return X, movie_ids

def aplicar_pca(X, varianza_objetivo=0.95):
    print(f"=== 2. Normalizar y aplicar PCA conservando el {int(varianza_objetivo*100)}% de la varianza ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=varianza_objetivo)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca

def guardar_resultados(X_pca, movie_ids):
    print("=== 3. Guardar resultados ===")

    # Guardar CSV
    df_pca = pd.DataFrame(X_pca)
    df_pca["movieId"] = movie_ids
    os.makedirs(os.path.dirname(PATH["output_csv"]), exist_ok=True)
    df_pca.to_csv(PATH["output_csv"], index=False)

    # Guardar .npy
    np.save(PATH["output_npy"], X_pca)
    np.save(PATH["output_ids_npy"], movie_ids)

def visualizar_componentes(X_pca):
    if X_pca.shape[1] < 2:
        print("⚠️ No se puede graficar en 2D: solo hay 1 componente retenida")
        return

    print("=== 4. Visualizar primeras 2 componentes ===")
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5, alpha=0.5)
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.title("Visualización PCA (2 primeras componentes)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PATH["fig_pca_2d"])
    plt.show()

def histograma_distancias(X, sample_size=500):
    print(f"=== 5. Histograma de distancias entre {sample_size} muestras ===")
    np.random.seed(42)
    sample_idx = np.random.choice(X.shape[0], size=sample_size, replace=False)
    X_sample = X[sample_idx]

    dist_matrix = pairwise_distances(X_sample)
    upper_tri_indices = np.triu_indices_from(dist_matrix, k=1)
    dists = dist_matrix[upper_tri_indices]

    plt.figure(figsize=(8, 5))
    plt.hist(dists, bins=50, color='steelblue', edgecolor='black', alpha=0.75)
    plt.title("Distribución de distancias euclidianas\nentre 500 muestras (PCA features)")
    plt.xlabel("Distancia euclidiana")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PATH["fig_hist"])
    plt.show()

if __name__ == "__main__":
    X, movie_ids = cargar_datos()
    X_pca, pca = aplicar_pca(X)
    guardar_resultados(X_pca, movie_ids)
    print(f"Componentes retenidos: {pca.n_components_} de {X.shape[1]} originales")
    print(f"Varianza explicada acumulada: {np.sum(pca.explained_variance_ratio_):.4f}")
    visualizar_componentes(X_pca)
    histograma_distancias(X)
