"""
TP06 — Tatouage Numerique par Patchwork
========================================
Auteur : Implementation pour ING-4-SSIRF
Cours  : Biometrie & Tatouage (Hamdi Chebbi)

Ce script implemente :
 - PARTIE 1 : Patchwork sur image en niveaux de gris
 - PARTIE 2 : Patchwork sur image RGB (canal R) + simulation d'attaques
            (bruit gaussien, compression JPEG, flou gaussien)
 - Comparaison rapide avec la methode LSB
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
OUT_DIR = "C:\\Users\\Dokka\\Desktop\\sem2\\TPS\\results_TP6"
os.makedirs(OUT_DIR, exist_ok=True)

KEY        = 12345     # Cle secrete partagee insertion / detection
N_PAIRS    = 10000     # Nombre de paires de pixels (A, B)
DELTA      = 5         # Force d'insertion du tatouage (offset applique)
THRESHOLD  = 1.0       # Seuil de decision sur la moyenne d_bar


# ============================================================
# ETAPE 1 : Image de test
# ============================================================
def build_test_image(size=512):
    """
    Construit une image de test synthetique (degrades + cercles + rectangles)
    pour avoir des textures variees, en l'absence d'image fournie.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Degrade horizontal sur le canal B
    for x in range(size):
        img[:, x, 0] = int(255 * x / size)
    # Degrade vertical sur le canal G
    for y in range(size):
        img[y, :, 1] = int(255 * y / size)
    # Texture sur le canal R
    xs, ys = np.meshgrid(np.arange(size), np.arange(size))
    img[:, :, 2] = ((np.sin(xs / 20.0) + np.cos(ys / 25.0)) * 60 + 128).astype(np.uint8)
    # Quelques formes
    cv2.circle(img, (150, 150), 70, (255, 255, 255), -1)
    cv2.rectangle(img, (300, 300), (450, 450), (50, 50, 50), -1)
    cv2.circle(img, (380, 130), 50, (200, 50, 100), -1)
    return img


# ============================================================
# PARTIE 1 — PATCHWORK NIVEAU DE GRIS
# ============================================================
def generate_pairs(shape, n_pairs, key):
    """
    Genere n_pairs paires de coordonnees (A_i, B_i) reproductibles a partir
    d'une cle secrete. Les deux ensembles A et B sont disjoints.
    """
    h, w = shape[:2]
    rng = np.random.default_rng(key)
    total = h * w
    # Tirage sans remise de 2*n_pairs indices
    indices = rng.choice(total, size=2 * n_pairs, replace=False)
    A_idx = indices[:n_pairs]
    B_idx = indices[n_pairs:]
    A = np.column_stack(np.unravel_index(A_idx, (h, w)))
    B = np.column_stack(np.unravel_index(B_idx, (h, w)))
    return A, B


def patchwork_embed(img, key=KEY, n_pairs=N_PAIRS, delta=DELTA):
    """
    Insertion Patchwork :
        a_i' = a_i + delta
        b_i' = b_i - delta
    Les valeurs sont saturees dans [0, 255].
    """
    wm = img.astype(np.int16).copy()
    A, B = generate_pairs(img.shape, n_pairs, key)
    for (ay, ax), (by, bx) in zip(A, B):
        wm[ay, ax] = wm[ay, ax] + delta
        wm[by, bx] = wm[by, bx] - delta
    wm = np.clip(wm, 0, 255).astype(np.uint8)
    return wm


def patchwork_detect(img, key=KEY, n_pairs=N_PAIRS, threshold=THRESHOLD):
    """
    Detection statistique :
        d_bar = (1/n) * sum (a_i - b_i)
    Sans tatouage  : E[d_bar] ~ 0
    Avec tatouage  : E[d_bar] ~ 2 * delta
    """
    A, B = generate_pairs(img.shape, n_pairs, key)
    a_vals = img[A[:, 0], A[:, 1]].astype(np.float64)
    b_vals = img[B[:, 0], B[:, 1]].astype(np.float64)
    d_bar = float(np.mean(a_vals - b_vals))
    detected = d_bar > threshold
    return d_bar, detected


# ============================================================
# Metriques de qualite
# ============================================================
def psnr(original, modified):
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255.0 ** 2) / mse)


# ============================================================
# Methode LSB (pour comparaison)
# ============================================================
def lsb_embed(img, message_bits, key=KEY):
    """Insere des bits dans le LSB de pixels choisis aleatoirement par la cle."""
    flat = img.flatten().copy()
    rng = np.random.default_rng(key)
    positions = rng.choice(flat.size, size=len(message_bits), replace=False)
    for pos, bit in zip(positions, message_bits):
        flat[pos] = (flat[pos] & 0xFE) | bit
    return flat.reshape(img.shape), positions


def lsb_extract(img, positions):
    flat = img.flatten()
    return [int(flat[p] & 1) for p in positions]


# ============================================================
# Attaques
# ============================================================
def add_noise(img, sigma=5):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def jpeg_compress(img, quality=50, path=None):
    if path is None:
        path = os.path.join(OUT_DIR, "tmp_compressed.jpg")
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def gaussian_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


# ============================================================
# PARTIE 1 : execution
# ============================================================
def run_part1(rgb_image):
    print("\n" + "=" * 60)
    print("PARTIE 1 - PATCHWORK NIVEAU DE GRIS")
    print("=" * 60)

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(OUT_DIR, "1_original_gray.png"), gray)

    # Detection AVANT insertion (controle negatif)
    d_before, det_before = patchwork_detect(gray)
    print(f"Detection AVANT insertion : d_bar = {d_before:+.4f}  -> tatouage detecte ? {det_before}")

    # Insertion
    watermarked = patchwork_embed(gray)
    cv2.imwrite(os.path.join(OUT_DIR, "1_watermarked_gray.png"), watermarked)

    # Detection APRES insertion (controle positif)
    d_after, det_after = patchwork_detect(watermarked)
    print(f"Detection APRES insertion : d_bar = {d_after:+.4f}  -> tatouage detecte ? {det_after}")
    print(f"Valeur theorique attendue : 2 * delta = {2 * DELTA}")

    # Qualite
    p = psnr(gray, watermarked)
    print(f"PSNR (original vs tatouee) : {p:.2f} dB")

    # Visualisation
    diff = cv2.absdiff(gray, watermarked)
    diff_amp = np.clip(diff.astype(np.int16) * 30, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Avant - Originale"); axes[0].axis("off")
    axes[1].imshow(watermarked, cmap="gray"); axes[1].set_title(f"Apres - Tatouee (PSNR={p:.1f} dB)")
    axes[1].axis("off")
    axes[2].imshow(diff_amp, cmap="hot"); axes[2].set_title("Difference (amplifiee x30)")
    axes[2].axis("off")
    plt.suptitle(f"Partie 1 : Patchwork grayscale | d_bar avant={d_before:+.3f}, apres={d_after:+.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "1_partie1_visualisation.png"), dpi=100, bbox_inches="tight")
    plt.close()

    return gray, watermarked, d_before, d_after


# ============================================================
# PARTIE 2 : execution
# ============================================================
def run_part2(rgb_image, gray_d_after):
    print("\n" + "=" * 60)
    print("PARTIE 2 - PATCHWORK RGB (canal R) + ATTAQUES")
    print("=" * 60)

    # OpenCV charge en BGR -> on extrait le canal Rouge (index 2)
    img_bgr = rgb_image.copy()
    R = img_bgr[:, :, 2].copy()

    # Detection avant
    d_before, _ = patchwork_detect(R)
    print(f"Canal R - Detection AVANT : d_bar = {d_before:+.4f}")

    # Insertion sur le canal R uniquement
    R_wm = patchwork_embed(R)
    img_wm = img_bgr.copy()
    img_wm[:, :, 2] = R_wm
    cv2.imwrite(os.path.join(OUT_DIR, "2_original_rgb.png"), img_bgr)
    cv2.imwrite(os.path.join(OUT_DIR, "2_watermarked_rgb.png"), img_wm)

    d_after, det_after = patchwork_detect(R_wm)
    print(f"Canal R - Detection APRES : d_bar = {d_after:+.4f}  -> detecte ? {det_after}")
    p_rgb = psnr(img_bgr, img_wm)
    print(f"PSNR RGB (image complete) : {p_rgb:.2f} dB")

    # Comparaison grayscale vs canal R
    print(f"\nComparaison d_bar :")
    print(f"  Grayscale : {gray_d_after:+.4f}")
    print(f"  Canal R   : {d_after:+.4f}")
    print("  -> Les deux donnent une moyenne proche de 2*delta ; le canal R")
    print("     concentre toute la deformation et reste invisible si delta est faible.")

    # ----- Attaques -----
    print("\n--- Robustesse face aux attaques (sur l'image RGB tatouee) ---")
    attacks = {
        "Aucune"           : img_wm,
        "Bruit gaussien sigma=5"  : add_noise(img_wm, sigma=5),
        "Bruit gaussien sigma=15" : add_noise(img_wm, sigma=15),
        "JPEG Q=75"        : jpeg_compress(img_wm, 75, os.path.join(OUT_DIR, "2_jpeg_q75.jpg")),
        "JPEG Q=50"        : jpeg_compress(img_wm, 50, os.path.join(OUT_DIR, "2_jpeg_q50.jpg")),
        "JPEG Q=25"        : jpeg_compress(img_wm, 25, os.path.join(OUT_DIR, "2_jpeg_q25.jpg")),
        "Flou gaussien 5x5": gaussian_blur(img_wm, 5),
        "Flou gaussien 7x7": gaussian_blur(img_wm, 7),
    }

    results = []
    print(f"\n{'Attaque':<28} {'d_bar':>10} {'Detecte':>10} {'PSNR (dB)':>12}")
    print("-" * 64)
    for name, attacked in attacks.items():
        Rch = attacked[:, :, 2]
        d, det = patchwork_detect(Rch)
        pa = psnr(img_bgr, attacked)
        results.append((name, d, det, pa))
        det_str = "OUI" if det else "NON"
        print(f"{name:<28} {d:>+10.4f} {det_str:>10} {pa:>12.2f}")

    # Visualisations attaques
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (name, attacked) in zip(axes.flat, attacks.items()):
        ax.imshow(cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB))
        d, det = patchwork_detect(attacked[:, :, 2])
        det_str = "OK" if det else "ECHEC"
        ax.set_title(f"{name}\nd_bar={d:+.2f} ({det_str})", fontsize=10)
        ax.axis("off")
    plt.suptitle("Partie 2 : Robustesse Patchwork (canal R) face aux attaques", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "2_partie2_attaques.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # Histogramme synthese
    names = [r[0] for r in results]
    dbars = [r[1] for r in results]
    colors = ["green" if r[2] else "red" for r in results]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(names, dbars, color=colors)
    ax.axvline(THRESHOLD, color="black", linestyle="--", label=f"seuil = {THRESHOLD}")
    ax.set_xlabel("d_bar (statistique de detection)")
    ax.set_title("Statistique de detection sous attaque (vert = detecte, rouge = manque)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "2_partie2_synthese.png"), dpi=100, bbox_inches="tight")
    plt.close()

    return results


# ============================================================
# Comparaison Patchwork vs LSB
# ============================================================
def run_comparison(rgb_image):
    print("\n" + "=" * 60)
    print("COMPARAISON PATCHWORK vs LSB")
    print("=" * 60)
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # Patchwork
    pw = patchwork_embed(gray)
    pw_attacked = jpeg_compress(pw, 50)
    d_pw_clean, _ = patchwork_detect(pw)
    d_pw_jpeg, _  = patchwork_detect(pw_attacked)

    # LSB
    rng = np.random.default_rng(KEY)
    bits = rng.integers(0, 2, size=N_PAIRS).tolist()
    lsb_img, positions = lsb_embed(gray, bits)
    lsb_recovered_clean = lsb_extract(lsb_img, positions)
    err_clean = sum(b != r for b, r in zip(bits, lsb_recovered_clean)) / len(bits)

    lsb_attacked = jpeg_compress(lsb_img, 50)
    lsb_recovered_jpeg = lsb_extract(lsb_attacked, positions)
    err_jpeg = sum(b != r for b, r in zip(bits, lsb_recovered_jpeg)) / len(bits)

    print(f"{'Methode':<12} {'Sans attaque':>20} {'JPEG Q=50':>20}")
    print("-" * 56)
    print(f"{'Patchwork':<12} {f'd_bar={d_pw_clean:+.2f}':>20} {f'd_bar={d_pw_jpeg:+.2f}':>20}")
    print(f"{'LSB':<12} {f'BER={err_clean*100:.1f}%':>20} {f'BER={err_jpeg*100:.1f}%':>20}")
    print("\nConclusion :")
    print("  - LSB : BER tres faible si pas d'attaque, mais s'effondre en JPEG.")
    print("  - Patchwork : moins precis (1 bit par image), mais robuste statistiquement.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    np.random.seed(0)
    img = build_test_image(512)
    cv2.imwrite(os.path.join(OUT_DIR, "0_test_image.png"), img)
    print("Image de test generee :", img.shape)

    gray_orig, gray_wm, d_b, d_a = run_part1(img)
    results = run_part2(img, d_a)
    run_comparison(img)

    print("\n" + "=" * 60)
    print("TOUS LES RESULTATS SONT DANS :", OUT_DIR)
    print("=" * 60)
