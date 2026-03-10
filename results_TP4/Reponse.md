# Réponses aux Questions d’Analyse

## 1. Pourquoi PCA nécessite un bon alignement des visages ?

La méthode **PCA (Eigenfaces)** compare les images **pixel par pixel**. Chaque image est transformée en un vecteur contenant toutes les valeurs des pixels.

Si les visages ne sont pas correctement alignés :

- les yeux peuvent être à des positions différentes
- la bouche peut être déplacée
- le nez peut être décalé

Dans ce cas, PCA considère ces différences comme des différences entre personnes.

Un mauvais alignement augmente donc la **distance entre deux images du même individu**, ce qui peut provoquer des erreurs de reconnaissance.

C’est pourquoi il est nécessaire de :

- détecter le visage
- centrer le visage
- redimensionner les images à la même taille

---

## 2. Que se passe-t-il si k (nombre de composantes principales) est trop faible ?

Si **k est trop petit**, PCA conserve très peu d’information.

Conséquences :

- une grande partie des détails du visage est perdue
- les visages deviennent très similaires
- la capacité de discrimination entre individus diminue

Donc :

- la reconnaissance devient moins précise
- plusieurs personnes différentes peuvent être confondues

En résumé :

> Trop peu de composantes → perte importante d'information.

---

## 3. Que se passe-t-il si k est trop élevé ?

Si **k est trop grand**, on conserve presque toutes les composantes.

Conséquences :

- on garde aussi le **bruit et les variations inutiles**
- le modèle devient **trop spécifique aux images d’apprentissage**

Cela peut provoquer :

- **surapprentissage (overfitting)**
- mauvaise généralisation sur de nouvelles images
- augmentation du temps de calcul

En résumé :

> Trop de composantes → modèle trop complexe et sensible au bruit.

---

## 4. Pourquoi la distance Euclidienne est adaptée dans l’espace PCA ?

Après projection PCA, chaque visage est représenté par un **vecteur de caractéristiques** dans un espace de dimension réduite.

Dans cet espace :

- les composantes principales sont **orthogonales**
- les données sont mieux séparées

La **distance Euclidienne** mesure la similarité entre deux vecteurs :

\[
d(x,y) = \sqrt{\sum_{i=1}^{k} (x_i - y_i)^2}
\]

Si la distance est :

- **petite → visages similaires**
- **grande → visages différents**

La distance Euclidienne est donc **simple et efficace** pour comparer les visages dans l’espace PCA.

---

## 5. Quelles sont les limites des Eigenfaces face aux variations d’illumination ?

Les Eigenfaces sont **sensibles aux conditions d’éclairage**.

Un changement de lumière peut modifier fortement les valeurs des pixels, même si c’est la **même personne**.

Problèmes possibles :

- ombres sur le visage
- lumière latérale
- forte luminosité
- variation de contraste

Dans ces cas, PCA peut considérer deux images de la **même personne comme différentes**.

Les Eigenfaces sont donc peu robustes face :

- aux variations d’illumination
- aux expressions faciales
- aux rotations du visage
- aux objets qui cachent le visage (lunettes, masque)

C’est pourquoi des méthodes plus avancées sont souvent utilisées aujourd’hui :

- **Fisherfaces**
- **LBPH**
- **Deep Learning (CNN)**

---

## Conclusion

La méthode **PCA (Eigenfaces)** permet de réduire la dimension des images et de représenter les visages dans un espace de caractéristiques compact.

Elle permet une reconnaissance relativement simple basée sur la distance entre vecteurs.

Cependant, elle présente certaines limites, notamment sa sensibilité à l’illumination, à l’alignement des visages et aux variations d’expression.