import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image, ImageFilter, ImageOps

Image.MAX_IMAGE_PIXELS = None 

def preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path)
    img = img.convert("L")
    img = img.resize((300, 300))
    img = ImageOps.equalize(img)
    img = img.point(lambda x: 255 if x > 128 else 0)
    img = img.filter(ImageFilter.FIND_EDGES)
    return np.array(img)


def compute_ssim(img1_path, img2_path):
    img1 = preprocess(img1_path)
    img2 = preprocess(img2_path)
    similarity = compare_ssim(img1, img2, data_range=255)
    return similarity, img1, img2


def decision(score, threshold=0.75):
    return "ACCEPTEE" if score >= threshold else "REJETEE"


img1 = "empreinte/e1.png"
img2 = "empreinte/e2.png"

score, i1, i2 = compute_ssim(img1, img2)

print("SSIM =", score)
print("Decision =", decision(score))

plt.subplot(121)
plt.imshow(i1, cmap="gray")
plt.title("Image 1")
plt.axis("off")

plt.subplot(122)
plt.imshow(i2, cmap="gray")
plt.title("Image 2")
plt.axis("off")

plt.show()
