import cv2
import numpy as np
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt

class FaceVerificationSystem:
    def __init__(self):
        """Initialisation du détecteur de visage Viola-Jones"""
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        self.reference_features = None
        self.reference_image = None
        
    def detect_face(self, image):
        """
        Détection de visage avec Viola-Jones
        Retourne les coordonnées du plus grand visage détecté
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Garder le plus grand visage
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return largest_face
    
    def extract_lbp_features(self, face_image):
        """
        Extraction des caractéristiques LBP
        Retourne l'histogramme normalisé (256 bins)
        """
        # Redimensionner à 128x128
        face_resized = cv2.resize(face_image, (128, 128))
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculer LBP
        lbp_image = self._compute_lbp(gray)
        
        # Histogramme normalisé
        hist = cv2.calcHist([lbp_image], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def _compute_lbp(self, image):
        """Calcul de la matrice LBP"""
        h, w = image.shape
        lbp_image = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = image[i, j]
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                # Codage binaire
                binary_pattern = ''.join([str(1 if n >= center else 0) for n in neighbors])
                lbp_image[i, j] = int(binary_pattern, 2)
        
        return lbp_image
    
    def setup_reference(self, image_path):
        """Enregistrement du visage de référence"""
        self.reference_image = cv2.imread(image_path)
        if self.reference_image is None:
            print("Erreur : impossible de charger l'image de référence")
            return False
        
        face = self.detect_face(self.reference_image)
        if face is None:
            print("Erreur : aucun visage détecté dans l'image de référence")
            return False
        
        x, y, w, h = face
        face_roi = self.reference_image[y:y+h, x:x+w]
        self.reference_features = self.extract_lbp_features(face_roi)
        print("Visage de référence enregistré avec succès")
        return True
    
    def verify_face(self, image_path, threshold=0.75):
        """
        Vérification du visage test
        Retourne : (similarity, decision, face_coords)
        """
        test_image = cv2.imread(image_path)
        if test_image is None:
            print("Erreur : impossible de charger l'image de test")
            return None, "ERROR", None
        
        if self.reference_features is None:
            print("Erreur : aucun visage de référence enregistré")
            return None, "ERROR", None
        
        face = self.detect_face(test_image)
        if face is None:
            print("Erreur : aucun visage détecté dans l'image de test")
            return None, "ERROR", None
        
        x, y, w, h = face
        face_roi = test_image[y:y+h, x:x+w]
        test_features = self.extract_lbp_features(face_roi)
        
        # Calcul de similarité
        distance = euclidean(self.reference_features, test_features)
        similarity = 1 - distance
        
        # Décision par seuillage
        decision = "Match" if similarity >= threshold else "No Match"
        
        return similarity, decision, (test_image, face, decision)


def main():
    """Programme principal"""
    reference_path = "faces/face1.jpg"
    test_path = "faces/face5.jpg"
    
    system = FaceVerificationSystem()
    
    if not system.setup_reference(reference_path):
        return
    
    similarity, decision, result = system.verify_face(test_path, threshold=0.75)
    
    if result is None:
        return
    
    print(f"\n{'='*50}")
    print(f"Similarité : {similarity:.4f}")
    print(f"Décision : {decision}")
    print(f"{'='*50}\n")
    
    test_image, (x, y, w, h), decision = result
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Image de référence
    ref_with_rect = system.reference_image.copy()
    ref_face = system.detect_face(system.reference_image)
    if ref_face is not None:
        x_, y_, w_, h_ = ref_face
        cv2.rectangle(ref_with_rect, (x_, y_), (x_+w_, y_+h_), (0, 255, 0), 2)
    axes[0].imshow(cv2.cvtColor(ref_with_rect, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Référence")
    axes[0].axis('off')
    
    # Image de test
    test_with_rect = test_image.copy()
    cv2.rectangle(test_with_rect, (x, y), (x+w, y+h), (0, 255, 0), 2)
    color = (0, 255, 0) if decision == "Match" else (0, 0, 255)
    cv2.putText(test_with_rect, decision, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    axes[1].imshow(cv2.cvtColor(test_with_rect, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Test - {decision} (Similarité: {similarity:.4f})")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()