
import os
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Carpeta donde guardaremos los posters subidos
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------------------------------
# Placeholder de datos: Reemplace esto con su propio dataset real.
# Cada película debe tener al menos: id, título, género, año, ruta del póster y (opcional) características numéricas.
# ---------------------------------------------------
movies = [
    {
        "id": 1,
        "title": "Ejemplo: Acción Extrema",
        "genre": "Action",
        "year": 2021,
        "poster_url": "/static/posters/accion_extrema.jpg",
        "features": None  # Aquí debería ir el vector de características visuales
    },
    {
        "id": 2,
        "title": "Ejemplo: Drama Intenso",
        "genre": "Drama",
        "year": 2019,
        "poster_url": "/static/posters/drama_intenso.jpg",
        "features": None
    },
    # … agregue tantas películas como desee …
]

# ---------------------------------------------------
# FUNCIONES AUXILIARES (placeholders)
# ---------------------------------------------------

def extract_image_features(image_path):
    """
    TODO: Implementar extracción de características visuales del póster (p. ej. con una CNN pre-entrenada).
    Por ahora, devuelve None o un vector dummy.
    """
    return None


def get_similar_movies(uploaded_features, k=5):
    """
    TODO: Compare 'uploaded_features' con el campo 'features' de cada película
    (por ejemplo, distancia Euclidiana o cosine similarity) y devuelva una lista
    de películas ordenadas de mayor a menor similitud. 
    Por ahora devolvemos las primeras k películas sin filtrar.
    """
    return movies[:k]


def filter_movies_by_metadata(genre=None, year=None):
    """
    Filtra la lista global 'movies' dejando solo las que coincidan con 'genre' y/o 'year'.
    """
    result = movies
    if genre:
        result = [m for m in result if m["genre"].lower() == genre.lower()]
    if year:
        try:
            y = int(year)
            result = [m for m in result if m["year"] == y]
        except ValueError:
            pass
    return result


def get_representative_per_cluster():
    """
    TODO: Implementar clustering (p. ej. KMeans sobre los vectores 'features'),
    luego para cada cluster elegir una película representativa (p. ej. la más
    cercana al centroide). Ahora devolvemos un placeholder con todas las películas.
    """
    # Ejemplo: asumimos que cada película es su propio clúster
    return movies


def generate_2d_plot():
    """
    TODO: Tome los vectores 'features' de todas las películas, aplique
    reducción a 2D (p. ej. TSNE o PCA) y genere un plot que se guarde en
    '/static/plot.png'. Por simplicidad devolvemos None y no generamos nada.
    """
    # from sklearn.manifold import TSNE
    # import matplotlib.pyplot as plt
    # X = [m["features"] for m in movies if m["features"] is not None]
    # tsne = TSNE(n_components=2)
    # coords = tsne.fit_transform(X)
    # plt.figure(figsize=(6,6))
    # plt.scatter(coords[:,0], coords[:,1])
    # plt.savefig("static/plot.png")
    return None


# ---------------------------------------------------
# RUTAS DE FLASK
# ---------------------------------------------------

@app.route('/')
def index():
    """
    Página principal: formulario para subir un póster y/o filtrar por género/año.
    """
    return render_template('index.html')


@app.route('/search_visual', methods=['POST'])
def search_visual():
    """
    Recibe un póster por formulario multipart, lo guarda, extrae sus características
    y busca películas similares.
    """
    file = request.files.get('poster')
    if not file:
        return redirect(url_for('index'))

    # Guardar archivo en disco
    filename = file.filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # Extraer features (placeholder)
    features = extract_image_features(save_path)

    # Obtener películas similares
    similares = get_similar_movies(features, k=10)
    return render_template('results.html', movies=similares, title="Similares al póster subido")


@app.route('/search_meta', methods=['GET'])
def search_meta():
    """
    Busca películas según los parámetros GET: género y/o año.
    """
    genre = request.args.get('genre', '').strip()
    year = request.args.get('year', '').strip()
    filtered = filter_movies_by_metadata(genre=genre or None, year=year or None)
    return render_template('results.html', movies=filtered, title="Resultados por filtro")


@app.route('/clusters')
def clusters():
    """
    Muestra una lista de películas representativas de cada clúster.
    """
    reps = get_representative_per_cluster()
    return render_template('clusters.html', movies=reps, title="Películas Representativas por Clúster")


@app.route('/plot')
def plot():
    """
    Genera o carga un gráfico 2D con la distribución de películas según sus features.
    """
    # Llamar a la función que genera el plot en 'static/plot.png'
    generate_2d_plot()
    return render_template('plot.html')  # en plot.html insertamos <img src="/static/plot.png">


if __name__ == '__main__':
    app.run(debug=True)