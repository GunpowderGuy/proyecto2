import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from skimage.util import img_as_ubyte

# === RUTA DE PÓSTERS ===
POSTER_DIR = "poster_ID"
files = [f for f in os.listdir(POSTER_DIR) if f.endswith(('.jpg', '.png'))]

# === CARGAR EXISTENTES ===
visual_csv = "visual_features.csv"
df_existing = pd.read_csv(visual_csv)
existing_ids = set(df_existing["movieId"].astype(str))

# === FILTRAR SOLO NUEVOS ===
new_files = [f for f in files if os.path.splitext(f)[0] not in existing_ids]

# === PARÁMETROS ===
lbp_radius = 1
lbp_n_points = 8 * lbp_radius
glcm_distances = [1]
glcm_angles = [0]
sift = cv2.SIFT_create() if hasattr(cv2, "SIFT_create") else None

feature_list = []

for filename in tqdm(new_files, desc="Procesando nuevos pósters"):
    movie_id = os.path.splitext(filename)[0]
    path = os.path.join(POSTER_DIR, filename)
    img = cv2.imread(path)

    if img is None:
        print(f"[x] No se pudo leer {filename}")
        continue

    img_resized = cv2.resize(img, (128, 192))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # HISTOGRAMAS RGB + HSV
    hist_r = cv2.calcHist([img_resized], [2], None, [32], [0, 256])
    hist_g = cv2.calcHist([img_resized], [1], None, [32], [0, 256])
    hist_b = cv2.calcHist([img_resized], [0], None, [32], [0, 256])
    hist_rgb = np.concatenate([hist_r, hist_g, hist_b]).flatten()
    hist_rgb /= np.sum(hist_rgb)

    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    hist_hsv = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    hist_hsv /= np.sum(hist_hsv)

    # LBP
    lbp = local_binary_pattern(gray, lbp_n_points, lbp_radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, lbp_n_points + 3), range=(0, lbp_n_points + 2))
    lbp_hist = lbp_hist.astype("float") / lbp_hist.sum()

    # GLCM
    gray_ubyte = img_as_ubyte(gray)
    glcm = graycomatrix(gray_ubyte, distances=glcm_distances, angles=glcm_angles, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    glcm_features = [contrast, dissimilarity, homogeneity, energy, correlation]

    # HOG
    hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, feature_vector=True)

    # SIFT
    if sift is not None:
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        sift_desc = np.mean(descriptors, axis=0) if descriptors is not None else np.zeros(128)
    else:
        sift_desc = np.zeros(128)

    # Final
    full_features = [movie_id] + \
                    hist_rgb.tolist() + \
                    hist_hsv.tolist() + \
                    lbp_hist.tolist() + \
                    glcm_features + \
                    hog_features.tolist() + \
                    sift_desc.tolist()

    feature_list.append(full_features)

# === NOMBRES DE COLUMNAS ===
n_rgb = 32 * 3
n_hsv = 32 * 3
n_lbp = lbp_n_points + 2
n_glcm = 5
n_hog = len(hog_features)
n_sift = 128

columns = ["movieId"] + \
          [f"rgb_{i}" for i in range(n_rgb)] + \
          [f"hsv_{i}" for i in range(n_hsv)] + \
          [f"lbp_{i}" for i in range(n_lbp)] + \
          ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy", "glcm_correlation"] + \
          [f"hog_{i}" for i in range(n_hog)] + \
          [f"sift_{i}" for i in range(n_sift)]

# === GUARDAR LOS NUEVOS Y UNIR ===
df_new = pd.DataFrame(feature_list, columns=columns)
df_combined = pd.concat([df_existing, df_new], ignore_index=True)
df_combined.to_csv(visual_csv, index=False)
print(f"✅ Agregadas {len(df_new)} nuevas filas. Total actual: {len(df_combined)}")
