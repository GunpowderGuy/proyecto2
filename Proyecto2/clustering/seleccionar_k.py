# clustering/seleccionar_k.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# === Paths ===
PATH = {
    "features_csv": "../reduccion/data_reduccion/features_umap.csv",
    "fig_elbow": "./fig_clustering/k_elbow.png",
    "fig_silhouette": "./fig_clustering/k_silhouette.png",
    "output_csv": "./data_clustering/metricas_kmeans.csv",
    "output_bestk_txt": "./data_clustering/mejor_k.txt"
}
os.makedirs("./fig_clustering", exist_ok=True)
os.makedirs("./data_clustering", exist_ok=True)

def evaluar_kmeans(X, k_range):
    registros = []

    for k in k_range:
        print(f"Evaluando k = {k} ...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertia = kmeans.inertia_

        if k > 1:
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = np.nan

        registros.append({"k": k, "inercia": inertia, "silhouette": silhouette})

    return pd.DataFrame(registros)

def graficar_resultados(df_metricas):
    plt.figure(figsize=(10, 4))

    # Elbow plot
    plt.subplot(1, 2, 1)
    plt.plot(df_metricas["k"], df_metricas["inercia"], marker='o')
    plt.title("Método del Codo")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Inercia")
    plt.grid(True)

    # Silhouette plot
    plt.subplot(1, 2, 2)
    plt.plot(df_metricas["k"], df_metricas["silhouette"], marker='o', color='orange')
    plt.title("Silhouette Score Promedio")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PATH["fig_elbow"])
    plt.savefig(PATH["fig_silhouette"])
    plt.close()
    print(f"✅ Figuras guardadas:\n - {PATH['fig_elbow']}\n - {PATH['fig_silhouette']}")

def guardar_k_optimo(k, path_txt):
    with open(path_txt, "w") as f:
        f.write(str(k))
    print(f"✅ Valor óptimo de k guardado en {path_txt}")

def main():
    df = pd.read_csv(PATH["features_csv"])
    X = df.drop(columns=["movieId"]).values
    k_range = list(range(2, 16))

    print("=== Evaluando valores de k ===")
    df_metricas = evaluar_kmeans(X, k_range)

    print("=== Guardando métricas ===")
    df_metricas.to_csv(PATH["output_csv"], index=False)
    print(f"✅ Archivo guardado: {PATH['output_csv']}")

    df_valid = df_metricas.dropna()
    best_row = df_valid.loc[df_valid["silhouette"].idxmax()]
    best_k = int(best_row["k"])
    best_score = best_row["silhouette"]
    print(f"\n🎯 Mejor valor de k según Silhouette Score: k = {best_k} (score = {best_score:.4f})")

    guardar_k_optimo(best_k, PATH["output_bestk_txt"])
    print("=== Graficando resultados ===")
    graficar_resultados(df_metricas)

if __name__ == "__main__":
    main()
