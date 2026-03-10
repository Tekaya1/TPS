from TP4 import FaceRecognitionPCA
import cv2


dataset_path = "dataset"

# choose a test image from dataset
test_image = "dataset/s1/1.pgm"


# ---------------------
# Charger dataset
# ---------------------
model = FaceRecognitionPCA(n_components=30)

X, y = model.load_dataset(dataset_path)

print("Dataset loaded :", X.shape)


# ---------------------
# Train PCA
# ---------------------
model.train(X, y)

print("PCA Model trained")


# ---------------------
# Test recognition
# ---------------------
label, distance, decision = model.recognize(test_image, threshold=3000)

print("Distance minimale :", distance)
print("Identité prédite :", label)
print("Décision :", decision)


# ---------------------
# Affichage image
# ---------------------
image = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)

image = cv2.resize(image,(400,400))

cv2.putText(image,
            f"Distance: {distance:.2f}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2)

cv2.putText(image,
            decision,
            (10,70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2)

cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()