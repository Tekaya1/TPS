## Partie 1
from PIL import Image
import matplotlib.pyplot as plt
import os

# os.makedirs("results", exist_ok=True)

img = Image.open("image.jpg")

# plt.figure()
# plt.subplot(111)
# plt.imshow(img)
# plt.axis("off")

# img.save("results/image_originale.png")


#  Partie 2
img_resize = img.resize((300, 200))

plt.figure()

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Originale")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_resize)
plt.title("Redimensionnée")
plt.axis("off")

img_resize.save("results/image_redimensionnee.png")

## Partie 3
# from PIL import ImageEnhance

# enhancer = ImageEnhance.Brightness(img)
# img_bright = enhancer.enhance(1.5)

# plt.figure()

# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.title("Originale")
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.imshow(img_bright)
# plt.title("Luminosité +")
# plt.axis("off")

# img_bright.save("results/image_luminosite_augmente.png")
# plt.show()

## Partie 4
# img_gray = img.convert("L")

# plt.figure()

# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.imshow(img_gray, cmap="gray")
# plt.axis("off")

# img_gray.save("results/image_gris.png")
# plt.show()

## Partie 5
# threshold = 128
# img_bin = img_gray.point(lambda x: 255 if x > threshold else 0)

# plt.figure()

# plt.subplot(1,2,1)
# plt.imshow(img_gray, cmap="gray")
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.imshow(img_bin, cmap="gray")
# plt.axis("off")

# img_bin.save("results/image_binarisee.png")
# plt.show()

## Partie 6
# from PIL import ImageFilter

# edges = img_gray.filter(ImageFilter.FIND_EDGES)

# plt.figure()

# plt.subplot(1,2,1)
# plt.imshow(img_gray, cmap="gray")
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.imshow(edges, cmap="gray")
# plt.axis("off")

# edges.save("results/image_contours.png")
# plt.show()

## Partie 7
# blur1 = img.filter(ImageFilter.GaussianBlur(1))
# blur2 = img.filter(ImageFilter.GaussianBlur(2))
# blur3 = img.filter(ImageFilter.GaussianBlur(3))

# plt.figure()

# plt.subplot(1,3,1)
# plt.imshow(blur1)
# plt.axis("off")

# plt.subplot(1,3,2)
# plt.imshow(blur2)
# plt.axis("off")

# plt.subplot(1,3,3)
# plt.imshow(blur3)
# plt.axis("off")

# blur3.save("results/image_flou_gaussien.png")
# plt.show()

## Partie 8
# hist = img_gray.histogram()

# plt.plot(hist)
# plt.title("Histogramme niveaux de gris")
# plt.show()

## partie 9
# from PIL import ImageOps

# equalized = ImageOps.equalize(img_gray)

# plt.figure()

# plt.subplot(1,2,1)
# plt.imshow(img_gray, cmap="gray")
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.imshow(equalized, cmap="gray")
# plt.axis("off")

# equalized.save("results/image_egalisee.png")
# plt.show()

