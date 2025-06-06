import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# UMAP y KMeans
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Para HSV + LBP
from skimage.feature import local_binary_pattern
from matplotlib.patches import Patch

# ---------------------- 1) CONFIGURACIÓN ----------------------

DIRECTORY_PATH   = r'C:\Users\axime\OneDrive\Documentos\proyecto 2 - ml\poster_ID'
MOVIES_CSV_PATH  = r'C:\Users\axime\OneDrive\Documentos\proyecto 2 - ml\movies.csv'
MAX_IMAGES       = 5000

# Parámetros HSV + LBP
HSV_BINS         = 64
LBP_RADIUS       = 1
LBP_N_POINTS     = 2 * LBP_RADIUS   # 2 vecinos → histograma 4 bins

# Parámetros UMAP
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1

# Parámetros K-Means
N_CLUSTERS       = 5   # Cambia a 6, 7, 8… según cuántos clusters quieras

# Diccionario de colores por género
GENRE_COLORS = {
    'Adventure': 'red',
    'Animation': 'green',
    'Children': 'blue',
    'Comedy': 'orange',
    'Fantasy': 'purple',
    'Drama': 'pink',
    'Action': 'cyan',
    'Romance': 'yellow'
}
DEFAULT_COLOR = 'gray'


# ---------------------- 2) CARGA DE RUTAS DE IMÁGENES ----------------------

all_files = os.listdir(DIRECTORY_PATH)
image_paths = [
    os.path.join(DIRECTORY_PATH, f)
    for f in all_files
    if f.lower().endswith('.jpg') or f.lower().endswith('.png')
]
image_paths = image_paths[:MAX_IMAGES]


# ---------------------- 3) CARGAR movies.csv ----------------------

movies_df = pd.read_csv(MOVIES_CSV_PATH)
movies_df['movieId'] = movies_df['movieId'].astype(int)
movies_df.set_index('movieId', inplace=True)


# ---------------------- 4) FUNCIONES AUXILIARES ----------------------

def load_and_preprocess(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (200, 200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def extract_hsv_lbp(img, gray):
    # 1) Histograma HSV (3 × HSV_BINS)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [HSV_BINS], [0, 256]).flatten()
    s = cv2.calcHist([hsv], [1], None, [HSV_BINS], [0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [HSV_BINS], [0, 256]).flatten()
    color_hist = np.concatenate([h, s, v]).astype('float32')
    # 2) LBP (2 vecinos → 4 bins)
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_N_POINTS + 3),
        range=(0, LBP_N_POINTS + 2)
    )
    lbp_hist = lbp_hist.astype('float32')
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return np.concatenate([color_hist, lbp_hist])  # dim = 3×64 + 4 = 196


# ---------------------- 5) PREPROCESAMIENTO + EXTRACCIÓN ----------------------

image_list, gray_list, movie_ids = [], [], []

print("⏳ Cargando y preprocesando imágenes…")
for p in tqdm(image_paths, desc="Preprocesando", unit="img"):
    img, gray = load_and_preprocess(p)
    image_list.append(img)
    gray_list.append(gray)
    movie_ids.append(int(os.path.splitext(os.path.basename(p))[0]))

print("⏳ Extrayendo características (HSV + LBP)…")
features_list = []
for img, gray in tqdm(zip(image_list, gray_list),
                      desc="Extrayendo features", total=len(image_list), unit="img"):
    feats = extract_hsv_lbp(img, gray)
    features_list.append(feats)

features_array = np.array(features_list, dtype='float32')  # (N, 196)


# ---------------------- 6) ESCALADO CON MINMAX ----------------------

# Dividimos en dos bloques: color_hist (192D) y lbp_hist (4D)
dim_color = 3 * HSV_BINS  # 192
dim_lbp   = LBP_N_POINTS + 2  # 4

color_block = features_array[:, :dim_color]
lbp_block   = features_array[:, dim_color: dim_color + dim_lbp]

scaler_color = MinMaxScaler()
scaler_lbp   = MinMaxScaler()

color_scaled = scaler_color.fit_transform(color_block)
lbp_scaled   = scaler_lbp.fit_transform(lbp_block)

X_scaled = np.hstack([color_scaled, lbp_scaled])  # (N, 196)


# ---------------------- 7) UMAP A 2D ----------------------

print("⏳ Ejecutando UMAP a 2D…")
umap = UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=2,
    random_state=42
)
X_umap = umap.fit_transform(X_scaled)  # (N, 2)


# ---------------------- 8) K-MEANS SOBRE UMAP ----------------------

print(f"⏳ Ejecutando K-Means con k={N_CLUSTERS}…")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(X_umap)
centroids_umap = kmeans.cluster_centers_  # (k, 2)


# ---------------------- 9) VISUALIZACIÓN 2D ----------------------

movies_df_filtered = movies_df.loc[movie_ids]

plt.figure(figsize=(12, 8))
for idx, mid in enumerate(movie_ids):
    genres = movies_df_filtered.loc[mid, 'genres'].split('|')
    main_genre = genres[0]
    color = GENRE_COLORS.get(main_genre, DEFAULT_COLOR)

    x, y = X_umap[idx, 0], X_umap[idx, 1]
    plt.scatter(x, y, c=color, s=30, alpha=0.6, edgecolor='k', linewidth=0.2)

# Dibujar centroides con “X” negras
plt.scatter(
    centroids_umap[:, 0], 
    centroids_umap[:, 1],
    s=200, c='black', marker='X', linewidths=2, label='Centroides'
)

plt.title(f'Clusters en 2D (bolitas) con UMAP + K-Means (k={N_CLUSTERS})', fontsize=14)
plt.xlabel('UMAP dimensión 1', fontsize=12)
plt.ylabel('UMAP dimensión 2', fontsize=12)
plt.grid(True)

# Leyenda
legend_elems = [Patch(facecolor=c, edgecolor='k', label=g) for g, c in GENRE_COLORS.items()]
legend_elems.append(Patch(facecolor=DEFAULT_COLOR, edgecolor='k', label='Other géneros'))
legend_elems.append(Patch(facecolor='black', edgecolor='k', label='Centroides'))
plt.legend(handles=legend_elems, title='Género principal y Centroides',
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# ---------------------- 10) IMPRIMIR TÍTULOS (HASTA 10) POR CLUSTER ----------------------

print("\nTítulos (hasta 10) por cluster K-Means (UMAP space):")
for c in np.unique(labels_km):
    print(f"\nCluster {c}:")
    idxs = np.where(labels_km == c)[0]
    sample_ids = [movie_ids[i] for i in idxs][:10]
    for mid in sample_ids:
        print(f"  • {mid}: {movies_df_filtered.loc[mid, 'title']}")
