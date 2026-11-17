# Langue des signes

Reconnaissance de l'alphabet de la langue des signes américaine (ASL) en direct avec une webcam.

## Ce qu'il faut

- Un PC sous Windows avec une webcam.
- Internet au premier lancement (installation et téléchargements automatiques).

## Installation (une seule fois)

Double-cliquer sur `installer.bat` et attendre la fin.

## Lancement

Double-cliquer sur `lancer.bat`. F11 passe en plein écran.

Les signes se font avec la **main gauche**, en entier dans l'image, à 40–60 cm de la caméra.

## Utilisation

- **1. Enregistrer des signes** : l'appli montre chaque lettre, on l'imite et elle l'enregistre.
- **2. Traduire en direct** : l'appli reconnaît les lettres et écrit le mot.
- **3. Mettre à jour et résultats** : l'appli apprend les signes enregistrés et affiche le score.

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `app.py` | L'application |
| `config.py` | Les réglages |
| `pretraitement.py` | Détection de la main par la couleur de peau |
| `mains.py` | Détection de la main avec MediaPipe |
| `reconnaissance.py` | Reconnaissance des lettres |
| `entrainer_points.py`, `entrainer_cnn.py` | Apprentissage des modèles |
| `outils.py` | Lecture et écriture des données |
| `references.py` | Images d'exemple des lettres |
| `live.py`, `figures.py` | Outils en ligne de commande |
