# clustering/comparar_gmm.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

# === RUTAS ===
PATH = {
    "features_csv": "../reduccion/data_reduccion/features_umap.csv",
    "labels_manual": "./data_clustering/gmm_resultados.csv",
    "fig_comparacion": "./fig_clustering/gmm_comparacion.png"
}

os.makedirs("./fig_clustering", exist_ok=True)

# === 1. CARGAR DATOS ===
df_features = pd.read_csv(PATH["features_csv"])
X = df_features.drop(columns=["movieId"]).values

df_manual = pd.read_csv(PATH["labels_manual"])
labels_manual = df_manual["cluster_gmm_manual"].values
movie_ids = df_manual["movieId"].values

print(f"✅ Datos cargados correctamente (shape: {X.shape})")

# === 2. APLICAR GMM DE SKLEARN ===
n_clusters = len(np.unique(labels_manual))
gmm_lib = GaussianMixture(n_components=n_clusters, random_state=42)
gmm_lib.fit(X)
labels_lib = gmm_lib.predict(X)

# === 3. COMPARAR RESULTADOS ===
ari = adjusted_rand_score(labels_manual, labels_lib)
print(f"\n🎯 ARI (Adjusted Rand Index) entre GMM manual y sklearn: {ari:.4f}")

# === 4. VISUALIZACIÓN COMPARATIVA ===
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].scatter(X[:, 0], X[:, 1], c=labels_manual, cmap='tab10', s=4, alpha=0.7)
axs[0].set_title("GMM Manual")

axs[1].scatter(X[:, 0], X[:, 1], c=labels_lib, cmap='tab10', s=4, alpha=0.7)
axs[1].set_title("GMM sklearn")

plt.suptitle(f"Comparación GMM Manual vs sklearn\nARI = {ari:.4f}")
plt.tight_layout()
plt.savefig(PATH["fig_comparacion"])
plt.close()

print(f"📊 Figura comparativa guardada en '{PATH['fig_comparacion']}'")

# === GUARDAR LABELS COMO .NPY PARA COMPATIBILIDAD GENERAL ===
output_labels_npy = "./data_clustering/labels_gmm_manual.npy"
np.save(output_labels_npy, labels_lib)
print(f"💾 Etiquetas guardadas en '{output_labels_npy}'")

