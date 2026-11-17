"""Détection des 21 points de la main avec MediaPipe, et calcul des caractéristiques.

MediaPipe sert de deuxième méthode, à comparer avec la chaîne classique
(pretraitement.py). Il donne aussi une boîte de main fiable pour le CNN.
"""
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config

# Liaisons entre les 21 points, pour dessiner le squelette de la main
LIAISONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
            (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def telecharger_modele():
    """Télécharge le modèle de MediaPipe la première fois (environ 8 Mo)."""
    if not config.FICHIER_MODELE_MAIN.exists():
        config.DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
        print("Téléchargement du modèle MediaPipe...")
        urllib.request.urlretrieve(config.URL_MODELE_MAIN, config.FICHIER_MODELE_MAIN)
    return str(config.FICHIER_MODELE_MAIN)


class DetecteurMain:
    """Utilisation :
        with DetecteurMain(video=True) as det:
            main = det.detecter(image_bgr, temps_ms)
    `main` vaut None si aucune main n'est trouvée, sinon un dictionnaire avec
    'points' (21 x 3, en pixels pour x et y), 'droite' (bool) et 'boite'.
    """

    def __init__(self, video=True, confiance=0.5):
        self.video = video
        self.mauvaise_main = False   # vrai si seule l'autre main (pas celle choisie) est vue
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=telecharger_modele()),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_hands=2,             # on cherche 2 mains puis on garde celle qui signe
            min_hand_detection_confidence=confiance,
            min_hand_presence_confidence=confiance,
            min_tracking_confidence=confiance)
        self.detecteur = vision.HandLandmarker.create_from_options(options)
        self.dernier_temps = -1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.fermer()

    def fermer(self):
        if self.detecteur is not None:
            self.detecteur.close()
            self.detecteur = None

    def detecter(self, image_bgr, temps_ms=0):
        """Renvoie la main qui signe (config.MAIN_UTILISEE), ou None."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.video:
            # MediaPipe exige des temps strictement croissants
            temps_ms = max(int(temps_ms), self.dernier_temps + 1)
            self.dernier_temps = temps_ms
            resultat = self.detecteur.detect_for_video(image, temps_ms)
        else:
            resultat = self.detecteur.detect(image)
        self.mauvaise_main = False
        if not resultat.hand_landmarks:
            return None
        h, w = image_bgr.shape[:2]
        voulue = config.MAIN_UTILISEE
        for points, cote in zip(resultat.hand_landmarks, resultat.handedness):
            # MediaPipe suppose une image de selfie (miroir), comme celle de l'appli
            droite = cote[0].category_name == "Right"
            if config.INVERSER_GAUCHE_DROITE:
                droite = not droite
            if voulue == "les deux" or (voulue == "droite") == droite:
                pts = np.array([[p.x * w, p.y * h, p.z * w] for p in points])
                return {"points": pts, "droite": droite,
                        "boite": boite_depuis_points(pts, w, h)}
        self.mauvaise_main = True    # une main est là, mais ce n'est pas la bonne
        return None


def boite_depuis_points(pts, w, h, marge=0.25):
    """Boîte carrée autour des 21 points, avec une marge."""
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cote = max(x2 - x1, y2 - y1) * (1 + 2 * marge)
    return (int(max(0, cx - cote / 2)), int(max(0, cy - cote / 2)),
            int(min(w, cx + cote / 2)), int(min(h, cy + cote / 2)))


def caracteristiques(main):
    """Vecteur de 63 nombres qui ne dépend ni de la position, ni de la taille de la main,
    ni de la main utilisée (gauche ou droite).

    - on prend le poignet (point 0) comme origine
    - on divise par la plus grande distance au poignet
    - pour une main gauche, on inverse x (comme dans un miroir)
    """
    pts = main["points"].copy()
    pts -= pts[0]
    if not main["droite"]:
        pts[:, 0] = -pts[:, 0]
    echelle = np.linalg.norm(pts[:, :2], axis=1).max()
    pts /= echelle + 1e-6
    return pts.flatten().astype(np.float32)


def dessiner(image, main, couleur=(0, 255, 0)):
    pts = main["points"].astype(int)
    for a, b in LIAISONS:
        cv2.line(image, tuple(pts[a, :2]), tuple(pts[b, :2]), couleur, 2)
    for p in pts:
        cv2.circle(image, tuple(p[:2]), 3, (0, 0, 255), -1)


# ---------------------------------------------------------------- gestes (J et Z)
# Points suivis pendant un geste : poignet (0), bout de l'index (8), bout de l'auriculaire (20).
# J : l'auriculaire dessine un crochet ; Z : l'index dessine un Z.
POINTS_SUIVIS = [0, 8, 20]


def _repere(main):
    """Origine (poignet), taille de la main et signe pour retourner une main gauche."""
    pts = main["points"]
    taille = np.linalg.norm(pts[:, :2] - pts[0, :2], axis=1).max() + 1e-6
    signe = 1.0 if main["droite"] else -1.0
    return pts[0, :2].copy(), taille, signe


def amplitude_mouvement(liste_mains):
    """Plus grand déplacement des points suivis depuis le début de la liste, en tailles de
    main. Une main immobile donne environ 0,05 ; un geste J ou Z donne plus de 0,5."""
    mains = [m for m in liste_mains if m is not None]
    if len(mains) < 2:
        return 0.0
    _, taille, _ = _repere(mains[0])
    depart = mains[0]["points"][POINTS_SUIVIS, :2]
    return max(float(np.linalg.norm(m["points"][POINTS_SUIVIS, :2] - depart, axis=1).max())
               for m in mains) / taille


def rogner(liste_mains, seuil=0.015):
    """Retire les moments immobiles au début et à la fin d'un geste : on garde de la
    première à la dernière image où un point suivi bouge de plus de `seuil` tailles de
    main d'une image à la suivante."""
    mains = [m for m in liste_mains if m is not None]
    if len(mains) < 3:
        return mains
    _, taille, _ = _repere(mains[0])
    pts = np.array([m["points"][POINTS_SUIVIS, :2] for m in mains])
    vitesse = np.linalg.norm(np.diff(pts, axis=0), axis=2).max(axis=1) / taille
    bouge = np.nonzero(vitesse > seuil)[0]
    if len(bouge) == 0:
        return mains
    return mains[max(0, bouge[0]):bouge[-1] + 2]


def caracteristiques_sequence(liste_mains, n=None):
    """Vecteur qui décrit un geste (environ 1 seconde de mains détectées).

    - on retire d'abord les moments immobiles (fonction rogner)
    - trajectoire des 3 points suivis, rééchantillonnée sur n instants, par rapport au
      poignet du premier instant et divisée par la taille de la main : n x 3 x 2 valeurs
    - forme de la main au milieu du geste (les 63 caractéristiques habituelles)
    """
    import config
    n = n or config.POINTS_GESTE
    mains = rogner(liste_mains)
    if len(mains) < 4:
        return None
    origine, taille, signe = _repere(mains[0])
    indices = np.linspace(0, len(mains) - 1, n).round().astype(int)
    traj = []
    for i in indices:
        p = (mains[i]["points"][POINTS_SUIVIS, :2] - origine) / taille
        p[:, 0] *= signe
        traj.append(p)
    forme = caracteristiques(mains[len(mains) // 2])
    return np.concatenate([np.array(traj).ravel(), forme]).astype(np.float32)
