from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
import io
import base64
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar embeddings y metadatos
df = pd.read_csv('../Proyecto2/clustering/data_clustering/movies_clustering_web.csv')

# Asegurarse de que tenemos las columnas necesarias
assert all(col in df.columns for col in ['movieId', 'title', 'genres', 'year', 'umap1', 'umap2', 'cluster_kmeans'])

# Preprocesar géneros
df['genres_list'] = df['genres'].str.split('|')

# Ruta principal
@app.route('/')
def index():
    genres = sorted(set(g for sub in df['genres_list'].dropna() for g in sub))
    years = sorted(df['year'].dropna().unique())
    methods = ['kmeans', 'dbscan', 'gmm']
    
    # Obtener películas representativas de cada cluster (kmeans)
    representative_movies = get_representative_movies('kmeans')
    
    return render_template('index.html', 
                         genres=genres, 
                         years=years, 
                         methods=methods,
                         representative_movies=representative_movies)

def get_representative_movies(method='kmeans', n=5):
    """Obtener películas más cercanas al centroide de cada cluster"""
    cluster_col = f'cluster_{method}'
    clusters = df[cluster_col].unique()
    representative_movies = {}
    
    for cluster in clusters:
        cluster_data = df[df[cluster_col] == cluster]
        center = cluster_data[['umap1', 'umap2']].mean().values
        dists = np.linalg.norm(cluster_data[['umap1', 'umap2']].values - center, axis=1)
        cluster_data = cluster_data.copy()
        cluster_data['distance_to_center'] = dists
        rep_movies = cluster_data.sort_values('distance_to_center').head(n)
        representative_movies[cluster] = rep_movies.to_dict('records')
    
    return representative_movies

# Ruta para servir posters
@app.route('/poster/<int:movie_id>')
def get_poster(movie_id):
    poster_path = f'../Proyecto2/data_MovieLens/poster_{movie_id}.jpg'
    if os.path.exists(poster_path):
        return send_from_directory('static', f'poster_{movie_id}.jpg')
    else:
        return send_from_directory('static', 'no_poster.jpg')

# Ruta para recomendaciones visuales
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    movie_id = int(data.get('movie_id'))
    n = int(data.get('n', 10))
    
    query = df[df['movieId'] == movie_id]
    if query.empty:
        return jsonify([])
    
    query_vec = query[['umap1', 'umap2']].values
    candidates = df[df['movieId'] != movie_id].copy()
    
    # Usar distancia coseno para mejor rendimiento con UMAP
    dists = cosine_similarity(query_vec, candidates[['umap1', 'umap2']])[0]
    candidates['similarity'] = dists
    top = candidates.sort_values('similarity', ascending=False).head(n)
    
    return jsonify(top.to_dict(orient='records'))

# Ruta para búsqueda por imagen
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path):
    """Extraer características básicas de la imagen (simplificado)"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 150))  # Tamaño similar a los posters
    return img.flatten()  # Esto es muy simplificado, en realidad deberías usar el mismo feature extractor que usaste para los posters

@app.route('/search_by_image', methods=['POST'])
def search_by_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extraer características (esto es un placeholder)
            query_features = extract_features(filepath)
            
            # Comparar con posters existentes (simplificado)
            # En una implementación real, deberías tener precomputados los features de los posters
            similarities = []
            for _, row in df.iterrows():
                poster_path = f'./static/poster_{row["movieId"]}.jpg'
                if os.path.exists(poster_path):
                    poster_features = extract_features(poster_path)
                    similarity = cosine_similarity([query_features], [poster_features])[0][0]
                    similarities.append(similarity)
                else:
                    similarities.append(0)
            
            df['similarity'] = similarities
            results = df.sort_values('similarity', ascending=False).head(10)
            
            # Convertir imagen a base64 para mostrarla
            with open(filepath, "rb") as img_file:
                encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
            
            return jsonify({
                'results': results.to_dict(orient='records'),
                'query_image': encoded_img
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            os.remove(filepath)  # Limpiar archivo temporal
    
    return jsonify({'error': 'Invalid file type'}), 400

# Ruta para obtener películas de un cluster específico
@app.route('/cluster', methods=['GET'])
def get_cluster():
    cluster_id = int(request.args['id'])
    method = request.args.get('method', 'kmeans')
    cluster_col = f'cluster_{method}'
    
    group = df[df[cluster_col] == cluster_id].sample(min(10, len(df[df[cluster_col] == cluster_id])))
    return jsonify(group.to_dict(orient='records'))

# Ruta para datos de visualización 2D con filtros
@app.route('/visualize', methods=['GET'])
def visualize():
    genre = request.args.get('genre')
    year = request.args.get('year')
    method = request.args.get('method', 'kmeans')
    cluster_col = f'cluster_{method}'
    
    data = df.copy()
    if genre:
        data = data[data['genres'].str.contains(genre, na=False)]
    if year:
        data = data[data['year'] == int(year)]
    
    return jsonify(
        data[['movieId', 'title', 'umap1', 'umap2', cluster_col, 'genres', 'year']]
        .rename(columns={cluster_col: 'cluster'})
        .to_dict(orient='records')
    )

# Ruta para autocompletado de búsqueda
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    
    results = df[df['title'].str.lower().str.contains(query)].head(10)
    return jsonify(results[['movieId', 'title']].to_dict(orient='records'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
