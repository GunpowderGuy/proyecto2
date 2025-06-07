# LDA.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import pairwise_distances
import os

# === PATHS CENTRADOS ===
PATH = {
    "features_csv"      : "../extraccion/data_extraccion/visual_features.csv",
    "movies_csv"        : "../data_MovieLens/movies.csv",
    "output_csv"        : "../reduccion/data_reduccion/features_lda.csv",
    "output_npy"        : "../reduccion/data_reduccion/features_lda.npy",
    "output_ids_npy"    : "../reduccion/data_reduccion/movie_ids_lda.npy",
    "fig_scatter"       : "./fig_reduccion/lda_2d_check.png",
    "fig_hist"          : "./fig_reduccion/lda_distance_distribution.png"
}

def cargar_datos():
    print("=== 1. Cargar datos ===")
    df_features = pd.read_csv(PATH["features_csv"])
    df_movies = pd.read_csv(PATH["movies_csv"])
    return df_features, df_movies

def preparar_etiquetas(df_movies):
    print("=== 2. Procesar etiquetas ===")
    df_movies['main_genre'] = df_movies['genres'].apply(
        lambda x: x.split('|')[0] if isinstance(x, str) and x != '(no genres listed)' else None
    )
    df_movies = df_movies.dropna(subset=['main_genre'])
    df_movies['genre_code'] = LabelEncoder().fit_transform(df_movies['main_genre'])
    return df_movies

def unir_features_etiquetas(df_features, df_movies):
    print("=== 3. Unir features con etiquetas ===")
    df_merged = pd.merge(df_features, df_movies[['movieId', 'genre_code', 'main_genre']], on='movieId')
    X = df_merged.drop(columns=['movieId', 'genre_code', 'main_genre']).values
    y = df_merged['genre_code'].values
    genres = df_merged['main_genre'].values
    movie_ids = df_merged['movieId'].values
    return X, y, genres, movie_ids

def aplicar_lda(X, y):
    print("=== 4. Aplicar LDA ===")
    X_scaled = StandardScaler().fit_transform(X)
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X_scaled, y)
    return X_lda, lda

def guardar_resultados(X_lda, movie_ids):
    print("=== 5. Guardar resultados ===")
    os.makedirs(os.path.dirname(PATH["output_csv"]), exist_ok=True)

    df_lda = pd.DataFrame(X_lda)
    df_lda["movieId"] = movie_ids
    df_lda.to_csv(PATH["output_csv"], index=False)

    np.save(PATH["output_npy"], X_lda)
    np.save(PATH["output_ids_npy"], movie_ids)

def visualizar_componentes(X_lda, genres):
    if X_lda.shape[1] >= 2:
        print("=== 6. Visualización 2D ===")
        lda_df = pd.DataFrame(X_lda[:, :2], columns=["LD1", "LD2"])
        lda_df["Género"] = genres

        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=lda_df, x="LD1", y="LD2", hue="Género", palette="tab20", s=20, alpha=0.6)
        plt.title("Visualización 2D con LDA (primeras 2 componentes)")
        plt.tight_layout()
        plt.savefig(PATH["fig_scatter"])
        plt.show()

def graficar_distancias(X_lda):
    print("=== 7. Gráfico de distribución de distancias ===")
    dist_matrix = pairwise_distances(X_lda)
    avg_distances = dist_matrix.mean(axis=1)

    plt.figure(figsize=(8, 5))
    sns.histplot(avg_distances, kde=True, bins=50, color='orange')
    plt.title("Distribución de distancias promedio entre películas (LDA)")
    plt.xlabel("Distancia promedio")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(PATH["fig_hist"])
    plt.show()

if __name__ == "__main__":
    df_features, df_movies = cargar_datos()
    df_movies = preparar_etiquetas(df_movies)
    X, y, genres, movie_ids = unir_features_etiquetas(df_features, df_movies)
    X_lda, lda = aplicar_lda(X, y)
    guardar_resultados(X_lda, movie_ids)
    print(f"Componentes retenidos: {X_lda.shape[1]}")
    visualizar_componentes(X_lda, genres)
    graficar_distancias(X_lda)
