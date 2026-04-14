"""
TP06 — Tatouage Numérique par LSB (Domaine Spatial)
Biométrie & Tatouage — ING-4-SSIRF

Implémentation complète :
  - Partie 1 : LSB Niveau de Gris
  - Partie 2 : LSB RGB
  - Partie 3 : LSB avec Clé Secrète
"""

import numpy as np
from PIL import Image
import random
import os

# ─────────────────────────────────────────────
# UTILITAIRES : Conversion texte ↔ binaire
# ─────────────────────────────────────────────

def text_to_bin(message: str) -> str:
    """Convertit un message texte en chaîne de bits (UTF-8)."""
    return ''.join(format(byte, '08b') for byte in message.encode('utf-8'))


def bin_to_text(bit_string: str) -> str:
    """Convertit une chaîne de bits en texte (UTF-8)."""
    # Découper en octets de 8 bits
    bytes_list = [bit_string[i:i+8] for i in range(0, len(bit_string), 8)]
    byte_values = bytes(int(b, 2) for b in bytes_list if len(b) == 8)
    return byte_values.decode('utf-8', errors='replace')


def get_bit_length(message: str) -> int:
    """Retourne le nombre de bits nécessaires pour encoder le message."""
    return len(message.encode('utf-8')) * 8


# ─────────────────────────────────────────────
# PARTIE 1 — LSB NIVEAU DE GRIS
# ─────────────────────────────────────────────

def embed_lsb_gray(image_path: str, message: str, output_path: str) -> None:
    """
    Tatouage LSB sur image en niveaux de gris.

    Étapes :
      1. Lire l'image en niveaux de gris
      2. Convertir le message en binaire
      3. Aplatir l'image (flatten)
      4. Modifier le LSB de chaque pixel concerné
      5. Reconstruire l'image
      6. Sauvegarder
    """
    # 1. Lire l'image en niveaux de gris
    img = Image.open(image_path).convert('L')
    pixels = np.array(img, dtype=np.uint8)
    original_shape = pixels.shape

    # 2. Convertir le message en binaire
    bits = text_to_bin(message)
    n_bits = len(bits)

    # 3. Aplatir l'image
    flat = pixels.flatten()

    # Vérification de capacité
    if n_bits > len(flat):
        raise ValueError(
            f"Message trop long : {n_bits} bits requis, "
            f"mais seulement {len(flat)} pixels disponibles."
        )

    # 4. Modifier le LSB des pixels
    for i, bit in enumerate(bits):
        # Mettre le LSB à 0 puis insérer le bit voulu
        flat[i] = (flat[i] & 0xFE) | int(bit)

    # 5. Reconstruire l'image
    stego = flat.reshape(original_shape)

    # 6. Sauvegarder
    Image.fromarray(stego, mode='L').save(output_path)
    print(f"[GRAY] Image tatouée sauvegardée → {output_path}")
    print(f"       Message encodé : {n_bits} bits dans {n_bits} pixels "
          f"({100 * n_bits / len(flat):.3f}% de l'image utilisée)")


def extract_lsb_gray(image_path: str, msg_len: int) -> str:
    """
    Extraction LSB depuis une image en niveaux de gris.

    Paramètres :
      image_path : chemin vers l'image tatouée
      msg_len    : longueur du message original (en caractères)

    Étapes :
      1. Lire l'image
      2. Extraire les LSB des premiers pixels
      3. Reconstituer le message
    """
    img = Image.open(image_path).convert('L')
    flat = np.array(img, dtype=np.uint8).flatten()

    n_bits = msg_len * 8  # 8 bits par caractère UTF-8 (ASCII)

    # Extraire les LSB
    bits = ''.join(str(pixel & 1) for pixel in flat[:n_bits])

    return bin_to_text(bits)


# ─────────────────────────────────────────────
# PARTIE 2 — LSB RGB
# ─────────────────────────────────────────────

def embed_lsb_rgb(image_path: str, message: str, output_path: str) -> None:
    """
    Tatouage LSB sur image couleur (RGB).
    Capacité x3 par rapport au mode gris.

    Étapes :
      1. Lire l'image couleur
      2. Convertir le message en binaire
      3. Parcourir les pixels (R, G, B) séquentiellement
      4. Insérer un bit dans le LSB de chaque canal
      5. Sauvegarder l'image tatouée
    """
    # 1. Lire l'image en RGB
    img = Image.open(image_path).convert('RGB')
    pixels = np.array(img, dtype=np.uint8)
    original_shape = pixels.shape  # (H, W, 3)

    # 2. Convertir le message en binaire
    bits = text_to_bin(message)
    n_bits = len(bits)

    # 3. Aplatir sur les 3 canaux → vecteur de longueur H*W*3
    flat = pixels.flatten()

    if n_bits > len(flat):
        raise ValueError(
            f"Message trop long : {n_bits} bits requis, "
            f"mais seulement {len(flat)} valeurs disponibles (H×W×3)."
        )

    # 4. Insérer les bits dans le LSB de chaque composante
    for i, bit in enumerate(bits):
        flat[i] = (flat[i] & 0xFE) | int(bit)

    # 5. Reconstruire et sauvegarder
    stego = flat.reshape(original_shape)
    Image.fromarray(stego, mode='RGB').save(output_path)
    print(f"[RGB]  Image tatouée sauvegardée → {output_path}")
    print(f"       Message encodé : {n_bits} bits dans {n_bits} composantes "
          f"({100 * n_bits / len(flat):.3f}% de la capacité utilisée)")


def extract_lsb_rgb(image_path: str, msg_len: int) -> str:
    """
    Extraction LSB depuis une image RGB.

    Paramètres :
      image_path : chemin vers l'image tatouée
      msg_len    : longueur du message original (en caractères)
    """
    img = Image.open(image_path).convert('RGB')
    flat = np.array(img, dtype=np.uint8).flatten()

    n_bits = msg_len * 8

    bits = ''.join(str(v & 1) for v in flat[:n_bits])

    return bin_to_text(bits)


# ─────────────────────────────────────────────
# PARTIE 3 — LSB AVEC CLÉ SECRÈTE
# ─────────────────────────────────────────────

def embed_lsb_key(image_path: str, message: str, output_path: str, key: int) -> None:
    """
    Tatouage LSB sécurisé par sélection pseudo-aléatoire de pixels.

    La clé (seed) génère un ordre de pixels unique — sans la clé,
    il est impossible de savoir où les bits sont cachés.

    Étapes :
      1. Lire l'image en niveaux de gris
      2. Convertir le message en binaire
      3. Générer des positions aléatoires avec seed = key (sans répétition)
      4. Insérer les bits dans les pixels aux positions sélectionnées
      5. Sauvegarder
    """
    img = Image.open(image_path).convert('L')
    pixels = np.array(img, dtype=np.uint8)
    original_shape = pixels.shape
    flat = pixels.flatten()

    bits = text_to_bin(message)
    n_bits = len(bits)

    if n_bits > len(flat):
        raise ValueError(
            f"Message trop long : {n_bits} bits requis, "
            f"seulement {len(flat)} pixels disponibles."
        )

    # 3. Générer n_bits positions uniques avec la clé comme seed
    rng = random.Random(key)
    positions = rng.sample(range(len(flat)), n_bits)

    # 4. Insérer les bits aux positions sélectionnées
    for pos, bit in zip(positions, bits):
        flat[pos] = (flat[pos] & 0xFE) | int(bit)

    stego = flat.reshape(original_shape)
    Image.fromarray(stego, mode='L').save(output_path)
    print(f"[KEY]  Image tatouée sauvegardée → {output_path}")
    print(f"       Clé utilisée : {key} | Positions pseudo-aléatoires : {n_bits} pixels")


def extract_lsb_key(image_path: str, msg_len: int, key: int) -> str:
    """
    Extraction LSB sécurisée avec clé secrète.

    Utilise exactement la même seed pour reproduire les positions.
    Une clé incorrecte retourne du texte aléatoire (aucun message lisible).
    """
    img = Image.open(image_path).convert('L')
    flat = np.array(img, dtype=np.uint8).flatten()

    n_bits = msg_len * 8

    rng = random.Random(key)
    positions = rng.sample(range(len(flat)), n_bits)

    bits = ''.join(str(flat[pos] & 1) for pos in positions)

    return bin_to_text(bits)


# ─────────────────────────────────────────────
# COMPARAISON VISUELLE — PSNR
# ─────────────────────────────────────────────

def compute_psnr(original_path: str, stego_path: str, mode: str = 'L') -> float:
    """
    Calcule le PSNR (Peak Signal-to-Noise Ratio) entre l'image originale
    et l'image tatouée. Un PSNR élevé (> 40 dB) indique une bonne invisibilité.
    """
    orig = np.array(Image.open(original_path).convert(mode), dtype=np.float64)
    stego = np.array(Image.open(stego_path).convert(mode), dtype=np.float64)

    mse = np.mean((orig - stego) ** 2)
    if mse == 0:
        return float('inf')

    psnr = 10 * np.log10((255.0 ** 2) / mse)
    return psnr


# ─────────────────────────────────────────────
# PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────

def main():
    message = "bonjour"

    # ── Dossier de sortie pour toutes les images générées ────
    output_dir = "lsb_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Dossier de sortie : ./{output_dir}/")

    input_image = os.path.join(output_dir, "input.png")

    # Vérification de l'image d'entrée
    if not os.path.exists(input_image):
        print(f"[INFO] Image '{input_image}' introuvable.")
        print("  → Création d'une image de test synthétique (256×256)...")
        gradient = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
        Image.fromarray(gradient, mode='L').save(input_image)
        print(f"  → Image '{input_image}' créée.")

    gray_out = os.path.join(output_dir, "gray_output.png")
    rgb_out  = os.path.join(output_dir, "rgb_output.png")
    key_out  = os.path.join(output_dir, "key_output.png")

    print("=" * 55)
    print("  TP06 — Tatouage Numérique LSB")
    print("=" * 55)

    # ── PARTIE 1 : LSB Gris ──────────────────────────────────
    print("\n[ PARTIE 1 — LSB Niveau de Gris ]")
    embed_lsb_gray(input_image, message, gray_out)
    extracted_gray = extract_lsb_gray(gray_out, len(message))
    print(f"       Message extrait  : '{extracted_gray}'")
    print(f"       Extraction OK    : {extracted_gray == message}")
    psnr_gray = compute_psnr(input_image, gray_out, mode='L')
    print(f"       PSNR             : {psnr_gray:.2f} dB")

    # ── PARTIE 2 : LSB RGB ───────────────────────────────────
    print("\n[ PARTIE 2 — LSB RGB ]")
    embed_lsb_rgb(input_image, message, rgb_out)
    extracted_rgb = extract_lsb_rgb(rgb_out, len(message))
    print(f"       Message extrait  : '{extracted_rgb}'")
    print(f"       Extraction OK    : {extracted_rgb == message}")
    psnr_rgb = compute_psnr(input_image, rgb_out, mode='RGB')
    print(f"       PSNR             : {psnr_rgb:.2f} dB")

    # ── PARTIE 3 : LSB avec Clé ──────────────────────────────
    print("\n[ PARTIE 3 — LSB avec Clé Secrète ]")
    embed_lsb_key(input_image, message, key_out, key=42)

    extracted_key_ok = extract_lsb_key(key_out, len(message), key=42)
    print(f"       Clé correcte (42) → '{extracted_key_ok}' | OK : {extracted_key_ok == message}")

    extracted_key_bad = extract_lsb_key(key_out, len(message), key=99)
    print(f"       Mauvaise clé (99) → '{extracted_key_bad}' | Correct : {extracted_key_bad == message}")

    psnr_key = compute_psnr(input_image, key_out, mode='L')
    print(f"       PSNR              : {psnr_key:.2f} dB")

    print(f"\n  Fichiers générés dans ./{output_dir}/ :")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"    • {f:<25} ({size:,} octets)")

    # ── COMPARAISON FINALE ───────────────────────────────────
    print("\n" + "=" * 55)
    print("  BILAN COMPARATIF")
    print("=" * 55)
    print(f"  {'Méthode':<25} {'PSNR (dB)':<15} {'Capacité'}")
    print(f"  {'-'*53}")

    img_tmp = Image.open(input_image).convert('L')
    w, h = img_tmp.size
    cap_gray = w * h
    cap_rgb  = w * h * 3
    cap_key  = w * h

    print(f"  {'LSB Gris':<25} {psnr_gray:<15.2f} {cap_gray} bits")
    print(f"  {'LSB RGB':<25} {psnr_rgb:<15.2f} {cap_rgb} bits (×3)")
    print(f"  {'LSB Clé secrète':<25} {psnr_key:<15.2f} {cap_key} bits (sécurisé)")
    print()
    print("  Notes :")
    print("  • PSNR > 40 dB → imperceptible à l'œil humain")
    print("  • La clé secrète n'améliore pas le PSNR mais rend")
    print("    l'extraction impossible sans connaître la clé.")
    print("=" * 55)


if __name__ == "__main__":
    main()