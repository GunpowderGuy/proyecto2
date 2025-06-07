# dbscan_manual.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import pairwise_distances
import seaborn as sns

# === RUTAS ===
PATH = {
    "features_csv": "../reduccion/data_reduccion/features_umap.csv",
    "movies_csv": "../data_MovieLens/movies.csv",
    "output_labels": "./data_clustering/labels_dbscan_manual.npy",
    "output_csv": "./data_clustering/dbscan_resultados.csv",
    "fig_clusters": "./fig_clustering/dbscan_manual.png"
}
os.makedirs("./data_clustering", exist_ok=True)
os.makedirs("./fig_clustering", exist_ok=True)

# === IMPLEMENTACIÓN DE DBSCAN DESDE CERO ===
def dbscan(X, eps=0.6, min_samples=5):
    n = X.shape[0]
    labels = np.full(n, -1, dtype=int)  # Inicializar todas las etiquetas como -1 (ruido)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    dists = pairwise_distances(X)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = np.where(dists[i] <= eps)[0]

        if len(neighbors) < min_samples:
            labels[i] = -1  # ruido
        else:
            labels[i] = cluster_id
            seeds = list(neighbors[neighbors != i])

            while seeds:
                j = seeds.pop()
                if not visited[j]:
                    visited[j] = True
                    j_neighbors = np.where(dists[j] <= eps)[0]
                    if len(j_neighbors) >= min_samples:
                        seeds += list(j_neighbors[~np.isin(j_neighbors, seeds)])

                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1
    return labels

# === VISUALIZACIÓN MEJORADA PARA DBSCAN ===
def visualizar_clusters(X, labels, path, title="DBSCAN Manual", ruido_label=-1):
    plt.figure(figsize=(8, 6))
    labels = np.array(labels)
    ruido = labels == ruido_label
    clusters = labels != ruido_label

    # Dibujar puntos por separado
    plt.scatter(X[ruido, 0], X[ruido, 1], c='lightgray', s=2, label='Ruido', alpha=0.5)
    plt.scatter(X[clusters, 0], X[clusters, 1], c=labels[clusters], cmap='tab10', s=3, alpha=0.7)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# === MAIN ===
def main():
    print("=== 1. Cargar datos UMAP reducidos ===")
    df = pd.read_csv(PATH["features_csv"])
    X = df.drop(columns=["movieId"]).values
    movie_ids = df["movieId"].tolist()

    print("=== 2. Ejecutar DBSCAN manual ===")
    labels = dbscan(X, eps=0.6, min_samples=5)
    np.save(PATH["output_labels"], labels)

    print("=== 3. Guardar resultados ===")
    df_result = pd.DataFrame(X, columns=["umap1", "umap2"])
    df_result["movieId"] = movie_ids
    df_result["cluster_dbscan_manual"] = labels
    df_result.to_csv(PATH["output_csv"], index=False)
    print(f"💾 Resultados guardados en '{PATH['output_csv']}'")

    print("=== 4. Visualizar ===")
    visualizar_clusters(X, labels, PATH["fig_clusters"], title="DBSCAN Manual")
    print(f"📊 Figura guardada en '{PATH['fig_clusters']}'")

    print("\nDistribución de puntos por cluster:")
    print(pd.Series(labels).value_counts().sort_index())

    try:
        print("\nEjemplos de títulos por cluster:")
        movies_df = pd.read_csv(PATH["movies_csv"], index_col="movieId")
        etiquetas_unicas = sorted(np.unique(labels[labels != -1]))  # Excluir -1
        for c in etiquetas_unicas:
            ids_cluster = df_result[df_result["cluster_dbscan_manual"] == c]["movieId"].tolist()
            muestra = ids_cluster[:10]
            print(f"\nCluster {c}:")
            for mid in muestra:
                title = movies_df.loc[mid, 'title'] if mid in movies_df.index else "(no disponible)"
                print(f"  • {mid}: {title}")
    except Exception as e:
        print(f"⚠️ No se pudo mostrar títulos por cluster: {e}")

if __name__ == "__main__":
    main()




