import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import os

# === Paths ===
PATH = {
    "original": "../extraccion/data_extraccion/visual_features.csv",
    "pca":      "./data_reduccion/features_pca.csv",
    "lda":      "./data_reduccion/features_lda.csv",
    "svd":      "./data_reduccion/features_svd.csv",
    "umap":     "./data_reduccion/features_umap.csv"
}
FIG_DIR = "./fig_comparacion"
os.makedirs(FIG_DIR, exist_ok=True)

def evaluar_tecnica(nombre, df_reducido, df_original):
    print(f"\n🔍 Evaluando {nombre.upper()}")

    if "movieId" not in df_reducido.columns:
        print("⚠️ No se encontró columna 'movieId'. Se omite esta técnica.")
        return

    # Alinear ambos datasets por movieId
    df_merged = df_reducido.merge(df_original, on="movieId", suffixes=("_red", "_orig"))
    X_reducido = df_merged.filter(like="_red").values
    X_orig_aligned = df_merged.filter(like="_orig").values

    # Calcular Trustworthiness
    try:
        trust = trustworthiness(X_orig_aligned, X_reducido, n_neighbors=15)
    except Exception as e:
        trust = np.nan
        print(f"⚠️ No se pudo calcular trustworthiness: {e}")

    # Calcular Silhouette Score usando KMeans
    try:
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_reducido)
        sil_score = silhouette_score(X_reducido, labels, metric="euclidean")
    except Exception as e:
        sil_score = np.nan
        print(f"⚠️ No se pudo calcular silhouette score: {e}")

    print(f" - Trustworthiness: {trust:.4f}" if not np.isnan(trust) else " - Trustworthiness: N/A")
    print(f" - Silhouette Score (k=10): {sil_score:.4f}" if not np.isnan(sil_score) else " - Silhouette Score: N/A")

    # Visualización
    if X_reducido.shape[1] >= 2:
        fig_path = os.path.join(FIG_DIR, f"reduc_{nombre.lower()}_2d.png")
        plt.figure(figsize=(8, 6))
        plt.scatter(X_reducido[:, 0], X_reducido[:, 1], c=labels, cmap='tab10', s=2, alpha=0.6)
        plt.title(f"{nombre.upper()} (primeras 2 componentes, KMeans k=10)")
        plt.xlabel(f"{nombre}-1")
        plt.ylabel(f"{nombre}-2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        print(f" - Figura guardada en {fig_path}")
    else:
        print("⚠️ No se pudo graficar: menos de 2 dimensiones")

def main():
    df_original = pd.read_csv(PATH["original"])
    feature_cols = df_original.columns.drop("movieId")
    df_original_renamed = df_original.rename(columns={col: f"{col}_orig" for col in feature_cols})

    for nombre, ruta in PATH.items():
        if nombre == "original":
            continue
        try:
            df = pd.read_csv(ruta)
            feature_cols = df.columns.drop("movieId")
            df_renamed = df.rename(columns={col: f"{col}_red" for col in feature_cols})
            evaluar_tecnica(nombre, df_renamed, df_original_renamed)
        except Exception as e:
            print(f"❌ Error cargando {nombre.upper()}: {e}")

if __name__ == '__main__':
    main()
