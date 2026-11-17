# Reconnaissance de l'alphabet ASL en direct

Trois méthodes, à comparer dans le rapport :

| Méthode | Trouver la main | Reconnaître la lettre |
|---|---|---|
| `cnn-peau` | chaîne classique : YCbCr, médian, morphologie, composante connexe (`pretraitement.py`) | CNN |
| `cnn-mp` | MediaPipe (boîte autour des 21 points) | CNN |
| `points` | MediaPipe (21 points) | kNN / SVM / MLP sur les points (`entrainer_points.py`) |

## Installation (Python 3.10 à 3.12 conseillé)

```
pip install -r requirements.txt
```

Le modèle de MediaPipe (8 Mo) se télécharge tout seul au premier lancement.
`modeles/cnn.keras` est déjà entraîné sur Sign MNIST (94 % sur son test).

## Utilisation

1. **Une seule fois par PC** : double-clic sur `installer.bat` (installe Python et les bibliothèques).
2. **Ensuite** : double-clic sur `lancer.bat`. Pas de nom à donner : chaque lancement est une
   « séance » anonyme. **On signe avec la main gauche** (réglable dans `config.py`,
   `MAIN_UTILISEE`). La fenêtre s'agrandit ; F11 ou le bouton « Plein écran » du menu.
   - **1. Enregistrer des signes** : l'appli montre le dessin de chaque lettre, on l'imite,
     espace, 3-2-1, elle enregistre. J et Z se font avec un geste (15 fois). La dernière étape,
     « autre mouvement », apprend ce qui n'est PAS un J ou un Z. « J'ai fini : mettre à jour »
     fait apprendre les signes.
   - **2. Traduire en direct** : l'appli reconnaît les lettres et écrit le mot en bas.
     Lettre fausse : garder le signe et taper la bonne lettre au clavier. J et Z : faire le
     geste puis arrêter la main. « Options avancées » (pour le rapport) : choisir le moteur
     de reconnaissance (3 boutons), voir les étapes du traitement, faire une capture.
   - **3. Mettre à jour et résultats** : réapprend tous les signes (quelques secondes) ou
     améliore le CNN (quelques minutes) et affiche le score « RÉSULTAT ». « Score mesuré
     sur : la dernière séance » = score honnête (séance jamais vue). Boutons pour ouvrir
     le dossier des résultats et celui des signes. « Effacer les signes de cette séance » et
     « Tout effacer » demandent un 2e clic pour confirmer.
     « Comparer les 3 moteurs » : une personne qui n'a rien enregistré fait les 24 lettres
     fixes ; les 3 moteurs analysent les mêmes images. Résultats dans `figures/` :
     `comparaison_moteurs_<date>.csv/.png`, `comparaison_details_<date>.csv` (image par
     image) et `confusion_test_<moteur>_<date>.png`.
   - Marche à suivre complète (qui enregistre, combien, quels tests) : `Feuille_de_route.docx`.
3. Plus il y a de mains différentes (amis, famille…), mieux l'appli reconnaît tout le monde.
4. Les 3 moteurs de reconnaissance :
   - Squelette de la main (par défaut) : MediaPipe place 21 points, la lettre est reconnue
     d'après leur position (kNN / SVM / MLP).
   - Photo de la main + CNN : MediaPipe trouve la main, un CNN regarde la photo 28 x 28.
   - Couleur de peau + CNN : traitement d'image classique (YCbCr, médian, morphologie),
     puis le CNN.
5. Données : `donnees/`. Modèles : `modeles/`. Résultats, figures et captures : `figures/`.
   Images d'exemple : `references/` (dessins de wpclipart.com, domaine public).

## Scripts en ligne de commande (pour aller plus loin)

- `python live.py --video essai.mp4 --sortie resultat.mp4` : direct sur une vidéo (lettres statiques)
- `python figures.py` : figure des étapes du traitement pour le rapport
- `python entrainer_points.py --test-personne hugo` : points + gestes
- `python entrainer_cnn.py` (Sign MNIST, CSV dans `sign_mnist/`) ou `--affiner --repartir`

## Fichiers

- `app.py` : l'application (menu, Apprendre, Direct avec correction, Entraîner)
- `config.py` : tous les réglages (seuils de peau, zone, lissage, stockage, gestes)
- `pretraitement.py` : chaîne de traitement d'image classique (YCbCr, médian, morphologie)
- `mains.py` : MediaPipe, caractéristiques des points et des gestes (J, Z)
- `reconnaissance.py` : les trois méthodes, les gestes et le lissage (vote, mot)
- `references.py` : images d'exemple des 26 lettres
- `entrainer_points.py`, `entrainer_cnn.py` : entraînement des modèles
- `outils.py` : lecture / écriture / suppression des données, matrices de confusion
- `live.py`, `figures.py` : direct sur vidéo, figures du rapport
