# extraccion/extraccion.py

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import local_binary_pattern

# ---------------------- CONFIGURACIÓN ----------------------

DIRECTORY_PATH   = '../data_MovieLens/poster_ID'
MOVIES_CSV_PATH  = '../data_MovieLens/movies.csv'
OUTPUT_FEATURES  = './data_extraccion/features_scaled.npy'
OUTPUT_IDS       = './data_extraccion/movie_ids.npy'
OUTPUT_CSV       = "./data_extraccion/visual_features.csv"

MAX_IMAGES       = 5000
HSV_BINS         = 64
LBP_RADIUS       = 1
LBP_N_POINTS     = 2 * LBP_RADIUS

# ---------------------- FUNCIONES AUXILIARES ----------------------

def load_and_preprocess(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (200, 200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def extract_hsv_lbp(img, gray):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [HSV_BINS], [0, 256]).flatten()
    s = cv2.calcHist([hsv], [1], None, [HSV_BINS], [0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [HSV_BINS], [0, 256]).flatten()
    color_hist = np.concatenate([h, s, v]).astype('float32')

    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_N_POINTS + 3),
        range=(0, LBP_N_POINTS + 2)
    )
    lbp_hist = lbp_hist.astype('float32')
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    return np.concatenate([color_hist, lbp_hist])


# ---------------------- PROCESAMIENTO Y EXTRACCIÓN ----------------------

def extraer_features():
    all_files = os.listdir(DIRECTORY_PATH)
    image_paths = [
        os.path.join(DIRECTORY_PATH, f)
        for f in all_files if f.lower().endswith(('.jpg', '.png'))
    ][:MAX_IMAGES]

    image_list, gray_list, movie_ids = [], [], []

    print("⏳ Cargando y preprocesando imágenes…")
    for p in tqdm(image_paths, desc="Preprocesando", unit="img"):
        img, gray = load_and_preprocess(p)
        image_list.append(img)
        gray_list.append(gray)
        movie_ids.append(int(os.path.splitext(os.path.basename(p))[0]))

    print("⏳ Extrayendo características (HSV + LBP)…")
    features_list = [extract_hsv_lbp(img, gray) for img, gray in tqdm(zip(image_list, gray_list), total=len(image_list), desc="Extrayendo features")]

    features_array = np.array(features_list, dtype='float32')

    dim_color = 3 * HSV_BINS
    dim_lbp   = LBP_N_POINTS + 2

    color_block = features_array[:, :dim_color]
    lbp_block   = features_array[:, dim_color: dim_color + dim_lbp]

    scaler_color = MinMaxScaler()
    scaler_lbp   = MinMaxScaler()

    color_scaled = scaler_color.fit_transform(color_block)
    lbp_scaled   = scaler_lbp.fit_transform(lbp_block)

    X_scaled = np.hstack([color_scaled, lbp_scaled])

    os.makedirs(os.path.dirname(OUTPUT_FEATURES), exist_ok=True)
    np.save(OUTPUT_FEATURES, X_scaled)
    np.save(OUTPUT_IDS, movie_ids)

    print(f"✅ Features escaladas guardadas en: {OUTPUT_FEATURES}")
    print(f"✅ movie_ids guardados en: {OUTPUT_IDS}")

    # === Guardar como .csv (visual_features.csv) ===
    df_visual = pd.DataFrame(X_scaled)
    df_visual["movieId"] = movie_ids
    
    df_visual.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ CSV combinado guardado en: {OUTPUT_CSV}")


if __name__ == '__main__':
    extraer_features()