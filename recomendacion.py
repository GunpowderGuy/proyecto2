import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# =========================================
# 0) CONFIGURACIÓN: RUTAS A TUS CSV
# =========================================

# Embeddings UMAP + etiquetas de cluster (k=4)
EMBEDDINGS_CSV = r'./Proyecto2/clustering/data_clustering/embeddings_with_kmcustom.csv'

# movies.csv con columnas al menos: movieId,title,genres
#MOVIES_CSV    = r'../data_MovieLens/movies.csv'
MOVIES_CSV    = r'./movies.csv'

# =========================================
# 1) CARGAR DATOS
# =========================================

# 1.1) Cargar embeddings + clusters. Esperamos columnas:
#      movieId | umap1 | umap2 | cluster_km_custom
emb_df = pd.read_csv(EMBEDDINGS_CSV, index_col='movieId')

# Verificar que las columnas existan
required_cols = {'umap1', 'umap2', 'cluster_km_custom'}
assert required_cols.issubset(set(emb_df.columns)), \
       f"Faltan columnas en {EMBEDDINGS_CSV}: {required_cols - set(emb_df.columns)}"

# Extraer matrices numpy de UMAP
# Nota: emb_df.index es la lista de movieId
X_umap = emb_df[['umap1', 'umap2']].values   # shape = (n_samples, 2)
clusters = emb_df['cluster_km_custom'].values # shape = (n_samples,)

# 1.2) Cargar movies.csv (para poder imprimir títulos)
movies_df = pd.read_csv(MOVIES_CSV, index_col='movieId')

# =========================================
# 2) FUNCIÓN PARA OBTENER LAS 10 PELÍCULAS MÁS SIMILARES
# =========================================

def get_similar_movies(movie_id: int,
                       emb_df: pd.DataFrame,
                       movies_df: pd.DataFrame,
                       n_neighbors: int = 10,
                       within_same_cluster: bool = True
                      ) -> pd.DataFrame:
    """
    Dado un movieId, devuelve un DataFrame con las n_neighbors películas más similares
    (según distancia euclídea en el espacio UMAP 2D).  
    
    Si within_same_cluster=True, solo busca entre las películas que comparten el mismo cluster que movie_id.
    Si within_same_cluster=False, busca entre todo el dataset.

    Retorna un DataFrame con columnas: ['movieId', 'title', 'genres', 'umap1', 'umap2', 'cluster', 'distance'].
    """

    if movie_id not in emb_df.index:
        raise ValueError(f"movieId {movie_id} no encontrado en embeddings.")

    # 2.1) Extraer índice de la fila correspondiente al movie_id
    #     emb_df.index es una lista de movieId, y queremos la posición en la matriz:
    all_movie_ids = emb_df.index.to_list()
    idx_input = all_movie_ids.index(movie_id)

    # 2.2) Coordenadas UMAP del movie_id
    coord_input = X_umap[idx_input].reshape(1, 2)  # shape = (1,2)

    # 2.3) Si queremos restringir a mismo cluster
    if within_same_cluster:
        cl_id = clusters[idx_input]
        mask_same = (clusters == cl_id)
        # Asegurarnos de incluir al menos un punto (él mismo)
        candidate_idxs = np.where(mask_same)[0]  # índices en [0..n_samples-1]
    else:
        candidate_idxs = np.arange(len(all_movie_ids))  # todos

    # 2.4) Construir submatriz de coordenadas UMAP de los candidatos (incluyendo el original)
    X_candidates = X_umap[candidate_idxs]  # shape = (n_cand, 2)

    # 2.5) Calcular distancia euclídea entre coord_input y todos los X_candidates
    #      cdist devuelve shape (1, n_cand)
    dists = cdist(coord_input, X_candidates, metric='euclidean').reshape(-1)

    # 2.6) Ordenar índices de candidatos por distancia ascendente
    #      Por defecto, el más cercano (dist=0) será él mismo, lo ignoraremos luego.
    sorted_idx_cand = np.argsort(dists)

    # 2.7) Recoger los n_neighbors+1 primeros (para luego excluir el mismo elemento)
    topk = sorted_idx_cand[: (n_neighbors + 1)]

    # 2.8) Eliminar el mismo movie_id de los resultados
    topk = [i for i in topk if candidate_idxs[i] != idx_input]
    topk = topk[:n_neighbors]  # quedarnos solo con n_neighbors reales

    # 2.9) Construir un DataFrame con la info de cada vecino
    neighbors = []
    for local_idx in topk:
        global_idx = candidate_idxs[local_idx]
        neighbor_id = all_movie_ids[global_idx]
        distance    = float(dists[local_idx])
        umap1, umap2 = X_umap[global_idx]
        clust        = int(clusters[global_idx])
        # Obtener título y géneros de movies_df
        title  = movies_df.loc[neighbor_id, 'title']  if neighbor_id in movies_df.index else ""
        genres = movies_df.loc[neighbor_id, 'genres'] if neighbor_id in movies_df.index else ""
        neighbors.append({
            'movieId': neighbor_id,
            'title': title,
            'genres': genres,
            'umap1': umap1,
            'umap2': umap2,
            'cluster': clust,
            'distance': distance
        })

    return pd.DataFrame(neighbors)


# =========================================
# 3) EJEMPLO DE USO
# =========================================

if __name__ == "__main__":
    # 3.1) Pedir al usuario un input (movieId)
    ejemplo_id = input("Ingresa un movieId para ver 10 similares: ").strip()
    if not ejemplo_id.isdigit():
        print("Debes escribir un ID numérico. Ejemplo válido: 1 o 1000.")
        exit(1)
    movie_id = int(ejemplo_id)

    # 3.2) Obtener los 10 vecinos dentro del mismo cluster
    print(f"\nPelículas 10 más similares a {movie_id} (mismo cluster):")
    try:
        df_similar_cluster = get_similar_movies(movie_id,
                                                emb_df,
                                                movies_df,
                                                n_neighbors=10,
                                                within_same_cluster=True)
        print(df_similar_cluster[['movieId','title','genres','distance']].to_string(index=False))
    except ValueError as e:
        print("  ⚠️", e)
        exit(1)

    # 3.3) Obtener los 10 vecinos en todo el espacio UMAP
    print(f"\nPelículas 10 más similares a {movie_id} (todo UMAP sin cluster):")
    df_similar_global = get_similar_movies(movie_id,
                                           emb_df,
                                           movies_df,
                                           n_neighbors=10,
                                           within_same_cluster=False)
    print(df_similar_global[['movieId','title','genres','distance']].to_string(index=False))

    print("\n¡Listo!")
