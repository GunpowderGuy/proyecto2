# comparar_dbscan.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import os

# === RUTAS ===
PATH = {
    "features_csv": "../reduccion/data_reduccion/features_umap.csv",
    "labels_manual": "./data_clustering/labels_dbscan_manual.npy",
    "figura": "./fig_clustering/dbscan_comparacion.png"
}
os.makedirs("./fig_clustering", exist_ok=True)

# === 1. Cargar datos reducidos UMAP ===
print("=== 1. Cargar datos UMAP ===")
df = pd.read_csv(PATH["features_csv"])
X = df.drop(columns=["movieId"]).values

# === 2. Cargar etiquetas del DBSCAN manual ===
print("=== 2. Cargar etiquetas manuales ===")
labels_manual = np.load(PATH["labels_manual"])

# === 3. Ejecutar DBSCAN con sklearn ===
print("=== 3. Ejecutar DBSCAN sklearn ===")
dbscan_sklearn = DBSCAN(eps=0.6, min_samples=5)
labels_sklearn = dbscan_sklearn.fit_predict(X)

# === 4. Calcular ARI ===
print("=== 4. Calcular ARI ===")
ari = adjusted_rand_score(labels_manual, labels_sklearn)
print(f"✅ ARI entre DBSCAN manual y sklearn: {ari:.4f}")

# === 5. Visualización comparativa ===
print("=== 5. Visualizar ===")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].scatter(X[:, 0], X[:, 1], c=labels_manual, cmap='tab10', s=3)
axs[0].set_title("DBSCAN Manual")

axs[1].scatter(X[:, 0], X[:, 1], c=labels_sklearn, cmap='tab10', s=3)
axs[1].set_title("DBSCAN sklearn")

plt.tight_layout()
plt.savefig(PATH["figura"])
plt.close()

print(f"📊 Figura comparativa guardada en {PATH['figura']}")
