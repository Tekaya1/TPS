import cv2
import numpy as np
import os


class FaceRecognitionPCA:

    def __init__(self, n_components=30):

        self.n_components = n_components

        # Viola-Jones face detector
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.mean = None
        self.eigenvectors = None
        self.projections = []
        self.labels = []


    # ------------------------------
    # Face Detection
    # ------------------------------
    def detect_face(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        if len(faces) == 0:
            return None

        # prendre le plus grand visage
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        (x, y, w, h) = faces[0]

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (100, 100))

        return face


    # ------------------------------
    # Load Dataset
    # ------------------------------
    def load_dataset(self, dataset_path):

        X = []
        y = []

        label = 0

        for person in os.listdir(dataset_path):

            person_path = os.path.join(dataset_path, person)

            if not os.path.isdir(person_path):
                continue

            for image_name in os.listdir(person_path):

                image_path = os.path.join(person_path, image_name)

                # read grayscale directly
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # resize to 100x100
                img = cv2.resize(img, (100, 100))

                face_vector = img.flatten()

                X.append(face_vector)
                y.append(label)

            label += 1

        X = np.array(X)
        y = np.array(y)

        return X, y


    # ------------------------------
    # PCA computation
    # ------------------------------
    def compute_pca(self, X):

        # mean face
        self.mean = np.mean(X, axis=0)

        # center data
        X_centered = X - self.mean

        # compute smaller covariance matrix
        L = np.dot(X_centered, X_centered.T)

        # eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # compute actual eigenfaces
        eigenfaces = np.dot(X_centered.T, eigenvectors)

        # normalize eigenfaces
        eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)

        # keep k components
        self.eigenvectors = eigenfaces[:, :self.n_components]


    # ------------------------------
    # Projection
    # ------------------------------
    def project(self, face_vector):

        centered = face_vector - self.mean

        projection = np.dot(centered, self.eigenvectors)

        return projection


    # ------------------------------
    # Train model
    # ------------------------------
    def train(self, X, y):

        self.compute_pca(X)

        for i in range(len(X)):

            proj = self.project(X[i])

            self.projections.append(proj)
            self.labels.append(y[i])


    # ------------------------------
    # Recognition
    # ------------------------------
    def recognize(self, image_path, threshold=3000):

        image = cv2.imread(image_path)

        face = self.detect_face(image)

        if face is None:
            return None, None, "No Face"

        face_vector = face.flatten()

        proj_test = self.project(face_vector)

        min_dist = float("inf")
        predicted_label = None

        for i, proj in enumerate(self.projections):

            dist = np.linalg.norm(proj_test - proj)

            if dist < min_dist:
                min_dist = dist
                predicted_label = self.labels[i]

        if min_dist < threshold:
            decision = "Match"
        else:
            decision = "No Match"

        return predicted_label, min_dist, decision