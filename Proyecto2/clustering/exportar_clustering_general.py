# exportar_clustering_general.py

import numpy as np
import pandas as pd
import os

# === RUTAS DE ENTRADA Y SALIDA ===
PATH = {
    "features_umap": "../reduccion/data_reduccion/features_umap.csv",
    "movies_csv": "../data_MovieLens/movies.csv",
    "labels_kmeans": "./data_clustering/labels_kmeans_manual.npy",
    "labels_dbscan": "./data_clustering/labels_dbscan_manual.npy",
    "labels_gmm": "./data_clustering/labels_gmm_manual.npy",
    "output_csv": "./data_clustering/movies_clustering_web.csv"
}

# === 1. CARGAR DATOS BASE ===
print("📥 Cargando datos reducidos y metadatos...")
df_umap = pd.read_csv(PATH["features_umap"])
df_movies = pd.read_csv(PATH["movies_csv"])
df_umap.rename(columns={"x": "umap1", "y": "umap2"}, inplace=True)
print("Columnas disponibles en df:", df_umap.columns.tolist())


# Unir metadatos
df = df_umap.merge(df_movies, on="movieId", how="left")

# === 2. CARGAR CLUSTERING ===
def cargar_cluster(path, nombre):
    try:
        labels = np.load(path)
        df[nombre] = labels
        print(f"✅ Etiquetas '{nombre}' cargadas: {np.unique(labels)}")
    except FileNotFoundError:
        df[nombre] = -1  # marcador de cluster desconocido
        print(f"⚠️ No se encontró archivo para '{nombre}', se asignará -1")

cargar_cluster(PATH["labels_kmeans"], "cluster_kmeans")
cargar_cluster(PATH["labels_dbscan"], "cluster_dbscan")
cargar_cluster(PATH["labels_gmm"], "cluster_gmm")

# === 3. PROCESAMIENTO ADICIONAL ===
# Extraer año desde el título si no hay campo 'year'
if 'year' not in df.columns:
    df['year'] = df['title'].str.extract(r"\((\d{4})\)").fillna(-1).astype(int)

# Ruta dummy para pósters (modifica si tienes carpeta real)
df["poster_url"] = df["movieId"].apply(lambda x: f"/static/posters/{int(x)}.jpg")

# Seleccionar columnas para la web
cols_final = ["movieId", "title", "genres", "year", "umap1", "umap2",
              "cluster_kmeans", "cluster_dbscan", "cluster_gmm", "poster_url"]
df_final = df[cols_final]

# === 4. EXPORTAR CSV ===
df_final.to_csv(PATH["output_csv"], index=False)
print(f"💾 Archivo generado en {PATH['output_csv']}")
