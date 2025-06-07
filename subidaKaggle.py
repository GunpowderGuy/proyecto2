import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

# Carga los embeddings con las columnas umap1, umap2 y movieId
df_embeddings = pd.read_csv('./data_clustering/embeddings_with_kmcustom.csv')

# Carga las películas de test, que son las queries
df_test = pd.read_csv('movies_test.csv')

# Verificar que 'movieId' está en ambos dataframes
assert 'movieId' in df_embeddings.columns, "Error: 'movieId' no está en embeddings"
assert 'movieId' in df_test.columns, "Error: 'movieId' no está en test"

# Definir columnas de embeddings
embedding_cols = ['umap1', 'umap2']

# Para guardar resultados
results = []

# Para cada movieId en test, encontrar sus recomendaciones
for query_movie_id in df_test['movieId']:
    # Obtener el vector embedding de la película consulta
    query_vector = df_embeddings.loc[df_embeddings['movieId'] == query_movie_id, embedding_cols].values

    if query_vector.size == 0:
        print(f"⚠️ movieId {query_movie_id} no tiene embedding. Se generan recomendaciones dummy.")
        for position in range(1, 11):
            results.append({
                'ID': f"{query_movie_id}_{position}",
                'query_movie_id': query_movie_id,
                'recommended_movie_id': -1,
                'position': position
            })
        continue

    # Seleccionar candidatos (todas excepto la película consulta)
    candidates = df_embeddings[df_embeddings['movieId'] != query_movie_id].copy()
    candidate_vectors = candidates[embedding_cols].values

    # Calcular distancia euclidiana
    distances = euclidean_distances(query_vector, candidate_vectors)[0]
    candidates['distance'] = distances

    # Top 10 más cercanas
    top_similars = candidates.sort_values('distance').head(10)

    # Agregar a resultados con el formato esperado
    for position, recommended_movie_id in enumerate(top_similars['movieId'], start=1):
        results.append({
            'ID': f"{query_movie_id}_{position}",
            'query_movie_id': query_movie_id,
            'recommended_movie_id': recommended_movie_id,
            'position': position
        })

# Crear dataframe final y guardar CSV
submission = pd.DataFrame(results)

# Validación: deben ser exactamente 29230 filas
assert len(submission) == 29230, f"❌ El archivo debe tener 29230 filas, pero tiene {len(submission)}"

submission.to_csv('submission.csv', index=False)
print("✅ ¡Archivo 'submission.csv' generado correctamente!")
