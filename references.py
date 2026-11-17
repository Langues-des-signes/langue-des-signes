"""Images d'exemple des 26 lettres de l'alphabet ASL.

Source : série « Sign language A.svg » ... « Sign language Z.svg » de Wikimedia Commons,
dessins de wpclipart.com, dans le DOMAINE PUBLIC (libres d'utilisation).
J et Z sont dessinés avec la flèche du mouvement.

Les images sont téléchargées une seule fois (environ 26 x 10 Ko) dans le dossier
references/, puis lues depuis ce dossier. Sans Internet, l'application affiche une
description écrite à la place.
"""
import threading
import urllib.request

import cv2
import numpy as np

import config

CREDIT = "Dessins : wpclipart.com, domaine public (Wikimedia Commons)"
_BASE = "https://thumb.wikimedia.org/wikipedia/commons/thumb/"
_CHEMINS = {
    "A": "2/27", "B": "1/18", "C": "e/e3", "D": "0/06", "E": "c/cd", "F": "8/8f",
    "G": "d/d9", "H": "9/97", "I": "1/10", "J": "b/b1", "K": "9/97", "L": "d/d2",
    "M": "c/c4", "N": "e/e6", "O": "e/e0", "P": "0/08", "Q": "3/34", "R": "3/3d",
    "S": "3/3f", "T": "1/13", "U": "7/7c", "V": "c/ca", "W": "8/83", "X": "b/b7",
    "Y": "1/1d", "Z": "0/0a",
}


def url(lettre, serveur=_BASE):
    nom = f"Sign_language_{lettre}.svg"
    return f"{serveur}{_CHEMINS[lettre]}/{nom}/250px-{nom}.png"


def fichier(lettre):
    return config.DOSSIER_REFERENCES / f"{lettre}.png"


def telecharger(log=print):
    """Télécharge les images manquantes. Renvoie le nombre d'images disponibles."""
    config.DOSSIER_REFERENCES.mkdir(exist_ok=True)
    manquantes = [l for l in config.ALPHABET if not fichier(l).exists()]
    echecs = 0
    for lettre in manquantes:
        if echecs >= 2:          # pas d'Internet : inutile d'insister pour chaque lettre
            log("Images d'exemple : pas de connexion, on réessaiera au prochain lancement.")
            break
        for serveur in [_BASE, "https://upload.wikimedia.org/wikipedia/commons/thumb/"]:
            try:
                requete = urllib.request.Request(
                    url(lettre, serveur),
                    headers={"User-Agent": "ProjetLangueDesSignes/1.0 (projet etudiant)"})
                with urllib.request.urlopen(requete, timeout=15) as r:
                    donnees = r.read()
                if donnees[:4] == b"\x89PNG":
                    fichier(lettre).write_bytes(donnees)
                    echecs = 0
                    break
                erreur = "réponse inattendue"
            except Exception as e:
                erreur = e
        else:
            echecs += 1
            log(f"Image de {lettre} non téléchargée ({erreur})")
    n = sum(fichier(l).exists() for l in config.ALPHABET)
    if manquantes:
        log(f"Images d'exemple : {n}/26 disponibles")
    return n


def telecharger_en_fond():
    """Lance le téléchargement sans bloquer l'application."""
    if all(fichier(l).exists() for l in config.ALPHABET):
        return None
    fil = threading.Thread(target=telecharger, daemon=True)
    fil.start()
    return fil


_cache = {}


def image(lettre, largeur, hauteur):
    """Image d'exemple (BGR, fond blanc) qui tient dans largeur x hauteur, ou None."""
    cle = (lettre, largeur, hauteur)
    if cle in _cache:
        return _cache[cle]
    if lettre not in _CHEMINS or not fichier(lettre).exists():
        return None   # pas encore téléchargée : on ne met pas None en cache
    img = cv2.imread(str(fichier(lettre)), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 4:   # fond transparent -> fond blanc
        alpha = img[:, :, 3:4].astype(np.float32) / 255
        img = (img[:, :, :3].astype(np.float32) * alpha + 255 * (1 - alpha)).astype(np.uint8)
    h, w = img.shape[:2]
    k = min(largeur / w, hauteur / h)
    img = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))),
                     interpolation=cv2.INTER_AREA)
    _cache[cle] = img
    return img


def coller(canevas, lettre, x, y, largeur, hauteur, fond=(255, 255, 255)):
    """Dessine l'image d'exemple centrée dans le rectangle (x, y, largeur, hauteur) sur un
    fond blanc. Renvoie False si l'image n'est pas disponible."""
    img = image(lettre, largeur - 10, hauteur - 10)
    if img is None:
        return False
    cv2.rectangle(canevas, (x, y), (x + largeur, y + hauteur), fond, -1)
    h, w = img.shape[:2]
    x0, y0 = x + (largeur - w) // 2, y + (hauteur - h) // 2
    canevas[y0:y0 + h, x0:x0 + w] = img
    return True
