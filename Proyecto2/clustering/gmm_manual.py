import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# === RUTAS ===
PATH = {
    "features_csv": "../reduccion/data_reduccion/features_umap.csv",
    "movies_csv": "../data_MovieLens/movies.csv",
    "output_csv": "./data_clustering/gmm_resultados.csv",
    "fig_clusters": "./fig_clustering/gmm_manual.png"
}

os.makedirs("./data_clustering", exist_ok=True)
os.makedirs("./fig_clustering", exist_ok=True)

class GMMImproved:
    def __init__(self, n_components=4, max_iter=100, tol=1e-4, random_state=42):
        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.weights = np.ones(self.k) / self.k
        self.means = X[np.random.choice(n_samples, self.k, replace=False)]
        self.covariances = np.array([np.cov(X.T) + 1e-6 * np.eye(n_features) for _ in range(self.k)])
        self.log_likelihood = []

        for iteration in range(self.max_iter):
            likelihood = np.zeros((n_samples, self.k))
            for i in range(self.k):
                diff = X - self.means[i]
                inv_cov = np.linalg.inv(self.covariances[i])
                det_cov = np.linalg.det(self.covariances[i])
                norm = np.sqrt((2 * np.pi) ** n_features * det_cov)
                likelihood[:, i] = self.weights[i] * np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=1)) / norm

            responsibilities = likelihood / np.sum(likelihood, axis=1, keepdims=True)
            Nk = responsibilities.sum(axis=0)
            self.weights = Nk / n_samples
            self.means = (responsibilities.T @ X) / Nk[:, None]
            self.covariances = np.array([
                ((responsibilities[:, i][:, None] * (X - self.means[i])).T @ (X - self.means[i])) / Nk[i] + 1e-6 * np.eye(n_features)
                for i in range(self.k)
            ])

            log_likelihood = np.sum(np.log(np.sum(likelihood, axis=1)))
            self.log_likelihood.append(log_likelihood)
            if iteration > 0 and abs(log_likelihood - self.log_likelihood[-2]) < self.tol:
                print(f"✅ GMM convergió en iteración {iteration}")
                break

    def predict(self, X):
        n_samples = X.shape[0]
        likelihood = np.zeros((n_samples, self.k))
        for i in range(self.k):
            diff = X - self.means[i]
            inv_cov = np.linalg.inv(self.covariances[i])
            det_cov = np.linalg.det(self.covariances[i])
            norm = np.sqrt((2 * np.pi) ** X.shape[1] * det_cov)
            likelihood[:, i] = self.weights[i] * np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=1)) / norm
        return np.argmax(likelihood, axis=1)

def main():
    print("=== 1. Cargar datos UMAP reducidos ===")
    df = pd.read_csv(PATH["features_csv"])
    X = df.drop(columns=["movieId"]).values
    movie_ids = df["movieId"].tolist()

    print("=== 2. Ejecutar GMM mejorado ===")
    gmm = GMMImproved(n_components=4, max_iter=100, tol=1e-4)
    gmm.fit(X)
    labels = gmm.predict(X)

    print("=== 3. Guardar resultados ===")
    df_result = pd.DataFrame(X, columns=["umap1", "umap2"])
    df_result["movieId"] = movie_ids
    df_result["cluster_gmm_manual"] = labels
    df_result.to_csv(PATH["output_csv"], index=False)
    print(f"💾 Resultados guardados en '{PATH['output_csv']}'")

    np.save("./data_clustering/labels_gmm_manual.npy", labels)

    print("=== 4. Visualizar ===")
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=3, alpha=0.7)
    plt.title("GMM Mejorado")
    plt.tight_layout()
    plt.savefig(PATH["fig_clusters"])
    plt.close()
    print(f"📊 Figura guardada en '{PATH['fig_clusters']}'")

    print("\nDistribución de puntos por cluster:")
    print(pd.Series(labels).value_counts().sort_index())

    try:
        print("\nEjemplos de títulos por cluster:")
        movies_df = pd.read_csv(PATH["movies_csv"], index_col="movieId")
        for c in sorted(np.unique(labels)):
            ids_cluster = df_result[df_result["cluster_gmm_manual"] == c]["movieId"].tolist()[:10]
            print(f"\nCluster {c}:")
            for mid in ids_cluster:
                title = movies_df.loc[mid, 'title'] if mid in movies_df.index else "(no disponible)"
                print(f"  • {mid}: {title}")
    except Exception as e:
        print(f"⚠️ No se pudo mostrar títulos por cluster: {e}")

if __name__ == "__main__":
    main()
