## Analyse du Résultat de Reconnaissance Faciale

### Résultat obtenu

| Paramètre           | Valeur |
|---------------------|--------|
| Identité détectée   | s2     |
| Distance            | 0.44   |
| Décision            | Match  |

### Interprétation

Le système de reconnaissance faciale basé sur **FaceNet** a identifié l'image test comme appartenant à la personne **s2**.

La distance calculée entre l'embedding du visage test et celui de la base de données est :

```
Distance = 0.44
```

Cette distance est **inférieure au seuil utilisé (0.8)**, ce qui signifie que les deux visages sont considérés comme appartenant à la **même personne**.

### Analyse

Dans les systèmes de reconnaissance faciale utilisant **FaceNet**, les distances typiques sont :

| Situation           | Distance typique |
|---------------------|------------------|
| Même image          | 0                |
| Même personne       | 0.4 – 0.7        |
| Personnes différentes | 0.9 – 1.4      |

Dans notre cas :

```
Distance = 0.44
```

Cette valeur se situe dans l'intervalle **0.4 – 0.7**, ce qui confirme que l'image test correspond bien à la personne **s2**.

### Conclusion

Le système a correctement reconnu l'identité du visage testé.  
La distance relativement faible entre les embeddings confirme une forte similarité entre les deux visages.

La décision finale est donc :

```
Match
```

Cela montre que l'approche basée sur **Deep Learning (FaceNet)** permet une reconnaissance faciale efficace en comparant les embeddings de visages.
