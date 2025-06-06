import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import local_binary_pattern, hog

# --------------------------- 1) CONFIGURACIÓN ---------------------------

# → Rutas (cambiarlas si es necesario)
DIRECTORY_PATH   = r'C:\Users\axime\OneDrive\Documentos\proyecto 2 - ml\poster_ID'
MOVIES_CSV_PATH  = r'C:\Users\axime\OneDrive\Documentos\proyecto 2 - ml\movies.csv'

# → Número de pósters a procesar (por batch)
N_POSTERS = 50000   # ajusta a 50000 

# → Parámetros HSV + LBP + HOG
HSV_BINS            = 64
LBP_RADIUS          = 1
LBP_N_POINTS        = 2 * LBP_RADIUS   # 2 vecinos → histograma 4 bins

HOG_PIXELS_PER_CELL = (32, 32)
HOG_CELLS_PER_BLOCK = (1, 1)
HOG_ORIENTATIONS    = 9

# → Escalado
SCALER_METHOD = 'minmax'  # 'minmax' o 'standard'

# → Parámetros UMAP
UMAP_N_NEIGHBORS = 10
UMAP_MIN_DIST    = 0.1

# → Nombres de salida
FEATURES_CSV        = 'features_20000.csv'
EMBEDDINGS_CSV      = 'embeddings_umap_20000.csv'
FEATURES_NPZ        = 'features_20000.npz'       # opción comprimida
EMBEDDINGS_NPZ      = 'embeddings_umap_20000.npz' # opción comprimida

# ---------------------- 2) LEER RUTAS DE IMÁGENES ----------------------

print("🗂 Preparando lista de rutas de imágenes…")
all_files = []
for root, dirs, files in os.walk(DIRECTORY_PATH):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_files.append(os.path.join(root, f))

image_paths = sorted(all_files)[:N_POSTERS]
n_images = len(image_paths)
print(f"→ Procesaremos {n_images} pósters.\n")

# ---------------------- 3) CARGAR movies.csv ----------------------

movies_df = pd.read_csv(MOVIES_CSV_PATH)
movies_df['movieId'] = movies_df['movieId'].astype(int)
movies_df.set_index('movieId', inplace=True)

# ---------------------- 4) FUNCIONES AUXILIARES ----------------------

def load_and_preprocess(path):
    """
    Carga una imagen, la redimensiona a 200×200 y la convierte a escala de grises.
    Devuelve (img_color, img_gray).
    """
    img = cv2.imread(path)
    img = cv2.resize(img, (200, 200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def extract_hsv_lbp_hog(img, gray):
    """
    Extrae un vector 1D float32 concatenando:
      1) Histograma HSV (3 × HSV_BINS)  
      2) LBP con 2 vecinos (4 bins)  
      3) HOG con celdas 32×32 y bloque 1×1  
    """
    # 1) Histograma HSV (192 dimensiones)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [HSV_BINS], [0, 256]).flatten()
    s = cv2.calcHist([hsv], [1], None, [HSV_BINS], [0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [HSV_BINS], [0, 256]).flatten()
    color_hist = np.concatenate([h, s, v]).astype('float32')  # dim = 3×HSV_BINS

    # 2) LBP (histograma de 4 bins)
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_N_POINTS + 3),
        range=(0, LBP_N_POINTS + 2)
    )
    lbp_hist = lbp_hist.astype('float32')
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # dim = 4

    # 3) HOG (celdas 32×32, bloque 1×1, 9 orientaciones)
    hog_desc = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=False,
        feature_vector=True
    ).astype('float32')
    # dim ~ ((200/32 - 1)*(200/32 - 1)*9) ≈ 180–200 (depende de la versión de skimage)

    return np.concatenate([color_hist, lbp_hist, hog_desc])

# ---------------------- 5) DETERMINAR DIMENSIÓN DEL DESCRIPTOR ----------------------

print("🔍 Calculando dimensión del descriptor con la primera imagen…")
img0, gray0 = load_and_preprocess(image_paths[0])
feat0 = extract_hsv_lbp_hog(img0, gray0)
n_features = feat0.shape[0]
print(f"→ Cada descriptor tendrá {n_features} dimensiones.\n")

# ---------------------- 6) EXTRACCIÓN DE FEATURES ----------------------

print("⏳ 1/3 Extrayendo features y guardando en memoria…")
features_array = np.zeros((n_images, n_features), dtype='float32')
movie_ids = []

for idx, path in enumerate(tqdm(image_paths, desc="Extrayendo", unit="img")):
    img, gray = load_and_preprocess(path)
    feats = extract_hsv_lbp_hog(img, gray)
    features_array[idx, :] = feats
    mid = int(os.path.splitext(os.path.basename(path))[0])
    movie_ids.append(mid)

print(f"→ Features extraídas con forma {features_array.shape}.\n")

# ---------------------- 7) ESCALADO POR BLOQUES ----------------------

print("⏳ 2/3 Escalando características por bloques…")
dim_color = 3 * HSV_BINS   # 192
dim_lbp   = LBP_N_POINTS + 2  # 4
dim_hog   = n_features - (dim_color + dim_lbp)

color_block = features_array[:, :dim_color]
lbp_block   = features_array[:, dim_color: dim_color + dim_lbp]
hog_block   = features_array[:, dim_color + dim_lbp: dim_color + dim_lbp + dim_hog]

if SCALER_METHOD == 'minmax':
    scaler_color = MinMaxScaler()
    scaler_lbp   = MinMaxScaler()
    scaler_hog   = MinMaxScaler()

    color_scaled = scaler_color.fit_transform(color_block)
    lbp_scaled   = scaler_lbp.fit_transform(lbp_block)
    hog_scaled   = scaler_hog.fit_transform(hog_block)
else:
    from sklearn.preprocessing import StandardScaler
    scaler_color = StandardScaler()
    scaler_lbp   = StandardScaler()
    scaler_hog   = StandardScaler()

    color_scaled = scaler_color.fit_transform(color_block)
    lbp_scaled   = scaler_lbp.fit_transform(lbp_block)
    hog_scaled   = scaler_hog.fit_transform(hog_block)

X_scaled = np.hstack([color_scaled, lbp_scaled, hog_scaled])
print(f"→ Dimensión tras escalado: {X_scaled.shape}.\n")

# ---------------------- 8) REDUCIR A 2D CON UMAP ----------------------

print("⏳ 3/3 Ejecutando UMAP a 2D (puede tardar unos minutos)…")
umap = UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=2,
    random_state=42
)
X_umap = umap.fit_transform(X_scaled)
print(f"→ Dimensión tras UMAP: {X_umap.shape} (n_images  2).\n")

# ---------------------- 9) GUARDAR BD DE FEATURES Y EMBEDDINGS ----------------------

print("💾 Guardando bases de datos para compartir…")

# 9.1) Guardar features escalados en un CSV (movieId + columnas f0..fN)
features_df = pd.DataFrame(
    X_scaled,
    index=movie_ids,
    columns=[f"f{i}" for i in range(n_features)]
)
features_df.index.name = 'movieId'
features_df.to_csv(FEATURES_CSV)
print(f"→ Features escalados guardados en '{FEATURES_CSV}' (shape {features_df.shape}).")

# También guardamos con compresión NumPy (.npz), por si se prefiere un formato binario
np.savez_compressed(
    FEATURES_NPZ,
    movieId=np.array(movie_ids, dtype=np.int32),
    features=X_scaled
)
print(f"→ Features escalados guardados en '{FEATURES_NPZ}' (formato comprimido).")

# 9.2) Guardar UMAP embeddings en un CSV (movieId, umap1, umap2)
emb_df = pd.DataFrame(
    X_umap,
    index=movie_ids,
    columns=['umap1', 'umap2']
)
emb_df.index.name = 'movieId'
emb_df.to_csv(EMBEDDINGS_CSV)
print(f"→ Embeddings UMAP guardados en '{EMBEDDINGS_CSV}' (shape {emb_df.shape}).")

# También guardamos en npz
np.savez_compressed(
    EMBEDDINGS_NPZ,
    movieId=np.array(movie_ids, dtype=np.int32),
    umap=X_umap
)
print(f"→ Embeddings UMAP guardados en '{EMBEDDINGS_NPZ}' (formato comprimido).")

print("\n✅ Bases de datos listas.")

