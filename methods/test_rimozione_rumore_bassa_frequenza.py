#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 19:40:03 2025

@author: leo
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import tifffile
import cv2

def enhance_hologram_contrast(img_matrix, method='percentile'):
    """
    Migliora il contrasto dell'ologramma (SOLO PER VISUALIZZAZIONE).
    Input: Matrice float (anche con valori negativi).
    Output: Matrice uint8 (0-255) pronta per essere salvata/plottata.
    """
    # Copia per non modificare l'originale
    img = img_matrix.copy()
    
    # 1. Gestione NaN/Infiniti (pulizia base)
    img = np.nan_to_num(img, nan=np.nanmean(img))

    if method == 'log':
        # La tua proposta: Shift + Log
        # Shiftiamo tutto sopra lo zero
        img_shifted = img - np.min(img) + 1.0
        # Logaritmo
        img_log = np.log(img_shifted)
        # Normalizziamo 0-255
        img_out = cv2.normalize(img_log, None, 0, 255, cv2.NORM_MINMAX)
        return img_out.astype('uint8')

    elif method == 'percentile':
        # Taglia i picchi di rumore (hot pixels)
        # vmin = 1° percentile, vmax = 99° percentile
        vmin, vmax = np.percentile(img, (1, 99))
        
        # Clippiamo i valori fuori dal range
        img_clipped = np.clip(img, vmin, vmax)
        
        # Normalizziamo 0-255
        img_out = (img_clipped - vmin) / (vmax - vmin) * 255.0
        return img_out.astype('uint8')

    elif method == 'clahe':
        # 1. Prima normalizziamo robustamente (percentile)
        vmin, vmax = np.percentile(img, (1, 99))
        img_clipped = np.clip(img, vmin, vmax)
        img_norm = (img_clipped - vmin) / (vmax - vmin) * 255.0
        img_uint8 = img_norm.astype('uint8')
        
        # 2. Applichiamo CLAHE
        # clipLimit: quanto contrasto forzare (2.0 - 4.0 è buono)
        # tileGridSize: grandezza delle aree locali (8x8 è standard)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img_uint8)
        return img_clahe

    else:
        # Normalizzazione standard Min-Max (sconsigliata con rumore termico)
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')


def SectorialRadialProfile(img, center=None, theta_min_deg=0, theta_max_deg=360, plot_debug=False):
    """
    Calcola il profilo radiale su un settore angolare specifico.
    Convenzione Cartesiana: 0=Est, 90=Nord, 180=Ovest (Antiorario).
    
    Args:
        img: Immagine 2D (array numpy).
        center: Tupla (cx, cy). Se None usa il centro immagine.
        theta_min_deg, theta_max_deg: Angoli del settore.
        plot_debug: Se True, mostra un grafico con l'immagine, il settore e il profilo.
    """
    
    rows, cols = img.shape
    
    # Gestione Centro
    if center is None:
        cx, cy = cols / 2.0, rows / 2.0
    else:
        cx, cy = center

    # --- 1. Calcolo Matematica (Come prima) ---
    y_grid, x_grid = np.indices((rows, cols))
    x_shifted = x_grid - cx
    y_shifted = y_grid - cy

    r_pixel = np.rint(np.sqrt(x_shifted**2 + y_shifted**2)).astype(np.int32)
    
    # Angolo (Convenzione Cartesiana: -y perché l'asse Y immagine va in giù)
    theta_rad = np.arctan2(-y_shifted, x_shifted) 
    theta_deg_img = np.degrees(theta_rad) % 360

    # Maschera Settore
    if theta_min_deg <= theta_max_deg:
        mask_angle = (theta_deg_img >= theta_min_deg) & (theta_deg_img <= theta_max_deg)
    else:
        mask_angle = (theta_deg_img >= theta_min_deg) | (theta_deg_img <= theta_max_deg)

    r_pixel = np.where(mask_angle, r_pixel, -1)

    # Appiattimento e Ordinamento
    sorted_idx = np.argsort(r_pixel.flat)
    sorted_r = r_pixel.flat[sorted_idx]
    sorted_img = img.flat[sorted_idx]

    valid_mask = sorted_r >= 0
    sorted_r = sorted_r[valid_mask]
    sorted_img = sorted_img[valid_mask]

    # Gestione caso vuoto
    if sorted_r.size == 0:
        print(" Nessun pixel trovato nel settore selezionato!")
        return np.array([]), np.array([])

    # Calcolo Media
    unique_r, unique_indices = np.unique(sorted_r, return_index=True)
    cumsum_img = np.cumsum(sorted_img, dtype=np.float64)
    cumsum_img = np.insert(cumsum_img, 0, 0)
    
    indices = np.append(unique_indices, sorted_r.size)
    sum_per_ring = cumsum_img[indices[1:]] - cumsum_img[indices[:-1]]
    count_per_ring = np.diff(indices)

    mean_intensity = sum_per_ring / count_per_ring

    # --- 2. PLOT DI DEBUG (Solo se richiesto) ---
    if plot_debug:
        plt.figure(figsize=(12, 5))
        
        # --- PANNELLO SINISTRO: Immagine e Settore ---
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray', origin='upper')
        plt.plot(cx, cy, 'rx', markersize=8, label='Centro') # Centro

        # Calcolo coordinate per disegnare le linee del settore
        # Lunghezza linea = diagonale immagine (per essere sicuri che esca)
        L = max(rows, cols)
        
        # Linea Start (Verde)
        rad_start = np.radians(theta_min_deg)
        # Nota il segno meno su Y: coordinate immagine vs cartesiane
        x_s = cx + L * np.cos(rad_start)
        y_s = cy - L * np.sin(rad_start)
        plt.plot([cx, x_s], [cy, y_s], 'g--', linewidth=2, label=f'Start {theta_min_deg}°')

        # Linea End (Magenta)
        rad_end = np.radians(theta_max_deg)
        x_e = cx + L * np.cos(rad_end)
        y_e = cy - L * np.sin(rad_end)
        plt.plot([cx, x_e], [cy, y_e], 'm--', linewidth=2, label=f'End {theta_max_deg}°')

        # Limiti assi per zoomare sull'immagine (altrimenti le linee vanno all'infinito)
        plt.xlim(0, cols)
        plt.ylim(rows, 0) # Invertito per le immagini
        plt.legend()
        plt.title(rf"Settore Analizzato: {theta_min_deg}° $\to$ {theta_max_deg}°")

        # --- PANNELLO DESTRO: Il Profilo Risultante ---
        plt.subplot(1, 2, 2)
        plt.plot(unique_r, mean_intensity, color='blue', linewidth=1.5)
        plt.grid(True, alpha=0.3)
        plt.xlabel("Raggio (pxl)")
        plt.ylabel("Intensità Media")
        plt.title("Profilo Radiale Settoriale")
        
        plt.tight_layout()
        plt.show()

    return unique_r, mean_intensity

def remove_background_keep_fringes(img, radius_block=5):
    """
    Rimuove il background (basse frequenze) mantenendo le frange.
    È un filtro PASSA-ALTO.
    """
    # 1. FFT
    f = fft2(img)
    fshift = fftshift(f)
    
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    
    # 2. Maschera PASSA-ALTO (High-Pass)
    # Partiamo da una maschera tutta BIANCA (1) -> Passa tutto
    mask = np.ones((rows, cols), dtype=np.uint8)
    
    # Creiamo un BUCO NERO (0) solo al centro
    y, x = np.ogrid[:rows, :cols]
    center_area = (x - ccol)**2 + (y - crow)**2 <= radius_block**2
    mask[center_area] = 0
    
    # 3. Applica Maschera
    fshift_filtered = fshift * mask
    
    # 4. IFFT
    img_back = ifft2(ifftshift(fshift_filtered))
    
    # Restituiamo il modulo
    return np.real(img_back), np.log(1 + np.abs(fshift))


def find_droplet_center_cv2(img_matrix, blur_size=51, invert=False, show_debug=True):
    """
    Versione DEBUG .
    """
    # Fix blur dispari
    if blur_size % 2 == 0: blur_size += 1

    # 1. Normalizza
    img_norm = cv2.normalize(img_matrix, None, 0, 255, cv2.NORM_MINMAX)
    img_8u = img_norm.astype('uint8')

    # 2. Blur
    blurred = cv2.GaussianBlur(img_8u, (blur_size, blur_size), 0)

    # 3. Soglia
    # Se invert=True (default): Goccia Bianca su sfondo Nero (CORRETTO per findContours)
    # Se invert=False: Goccia Nera su sfondo Bianco (SBAGLIATO, troverebbe lo sfondo)
    threshold_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if invert else cv2.THRESH_BINARY + cv2.THRESH_OTSU
    _, thresh = cv2.threshold(blurred, 0, 255, threshold_type)

    # 4. Trova Contorni
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- PREPARIAMO L'IMMAGINE PER IL DISEGNO (Conversione in colori per vedere le linee) ---
    debug_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    if not contours:
        print("ERRORE: Nessun contorno trovato! (L'immagine è tutta nera?)")
        if show_debug:
            plt.imshow(thresh, cmap='gray')
            plt.title("Soglia (Tutto Nero?)")
            plt.show()
        return img_matrix.shape[1] // 2, img_matrix.shape[0] // 2

    # Disegna TUTTI i contorni in VERDE sottile
    cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 1)

    # Trova il più grande
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    print(f"Area del contorno più grande: {area} pixel")

    # Disegna il contorno "vincitore" in ROSSO spesso
    cv2.drawContours(debug_img, [largest_contour], -1, (255, 0, 0), 3)
    
    
    # 5. Momenti
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print(f"Centro calcolato dai momenti: {cX}, {cY}")
    else:
        print("ERRORE: Il contorno ha area zero o momento nullo!")
        cX, cY = img_matrix.shape[1] // 2, img_matrix.shape[0] // 2
    
    ellipse = cv2.fitEllipse(largest_contour)
    (xe, ye), (d1, d2), angle = ellipse
    radius_inscribed = min(d1, d2) / 2
    
    cv2.ellipse(img_8u, ellipse, (0, 255, 255), 2)
    
    center_int = (int(xe), int(ye))
    radius_int = int(radius_inscribed)
    
    cv2.circle(img_8u, center_int, radius_int, (0, 0, 255), 3)
    
    # Disegna la X del centro
    cv2.drawMarker(img_8u, (cX, cY), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # --- DEBUG VISIVO ---
    if show_debug:
        fig, ax = plt.subplots(1,2)
        
        ax[0].imshow(img_8u, cmap='gray')
        ax[0].set_title("Originale")
    
        
        ax[1].imshow(debug_img)
        ax[1].set_title(f"Contorni (Verde=Tutti, Rosso=Max)\nArea: {area}")
        ax[1].plot(cX,cY,'r+')
        
        plt.tight_layout()
        plt.show()

    return cX, cY



def radial_profile(data, center=None):
    """
    Calcola la media radiale.
    """
    y, x = np.indices((data.shape))
    
    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
        
    # Calcola la distanza di ogni pixel dal centro
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    
    # Somma i pixel per ogni raggio e conta quanti sono
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    
    # Media
    radialprofile = tbin / nr
    return radialprofile

# --- ESECUZIONE ---
img_path = "/home/leo/Scrivania/fiji_images/ologramma.tif"
img = tifffile.imread(img_path)


img_clean, spectrum = remove_background_keep_fringes(img, radius_block=10)

# --- PLOT ---
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Originale (Con macchie)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_clean, cmap='gray')
plt.title("Background Rimosso (Frange Salve!)")
plt.axis('off')

plt.tight_layout()
plt.show()



# --- VERIFICA ISTOGRAMMA ---
print(f"Min: {img_clean.min():.2f}, Max: {img_clean.max():.2f}, Mean: {img_clean.mean():.2f}")
# La media dovrebbe essere vicina a ZERO.



# --- USO ---
Cx, Cy = find_droplet_center_cv2(img, blur_size=400, show_debug=True)


# profilo radiale medio su tutto langolo 360 gradi
prof_radiale = radial_profile(img, center = (Cx,Cy))

plt.figure(figsize=(10,4))
plt.plot(prof_radiale)
plt.title("Profilo Radiale (Media sugli anelli)")
plt.xlabel("Distanza dal centro (pixel)")
plt.ylabel("Intensità Media")
plt.grid(True, alpha=0.3)
plt.show()


# sectorial angular average
r_full, i_full = SectorialRadialProfile(img, center=(Cx,Cy), theta_min_deg=0, theta_max_deg=360, plot_debug = True)

r_right, i_right = SectorialRadialProfile(img, center=(Cx,Cy),theta_min_deg=-45, theta_max_deg=45, plot_debug = True)

r_right, i_right = SectorialRadialProfile(img, center=(Cx,Cy),theta_min_deg=-5, theta_max_deg=5, plot_debug = True)


# --- 3. PLOT ---
plt.figure(figsize=(10, 6))

# Plot dei profili
plt.plot(r_full, i_full, label='Media su 360° (Full)', color='black', linewidth=2)
plt.plot(r_right, i_right, label='Media settore Destro (-45° a 45°)', color='red', linestyle='--')

plt.title("Confronto Profili Radiali")
plt.xlabel("Distanza dal centro (pixel)")
plt.ylabel("Intensità Media")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



h_profile = img[Cy, :]
v_profile = img[:, Cx]

# Plot
plt.plot(h_profile)
plt.title("Profilo Lineare (Sezione orizzontale)")
plt.show()

plt.plot(v_profile)
plt.title("Profilo Lineare (Sezione verticale)")
plt.show()



rect_profile = np.mean(img, axis=0)

plt.plot(rect_profile)
plt.title("Profilo Integrato (Media Verticale)")
plt.show()
