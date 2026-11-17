"""Chaîne de traitement d'image « classique » : trouver la main sans réseau de neurones.

Étapes (chacune est gardée dans un dictionnaire pour les figures du rapport) :
  1. zone d'intérêt (on ne garde que la partie de l'image où on signe)
  2. passage en YCbCr et seuillage de Cb et Cr  -> masque de peau
  3. (option) soustraction d'un fond appris      -> masque de mouvement
  4. filtre médian                                -> enlève le bruit poivre et sel
  5. ouverture puis fermeture morphologiques      -> enlève les petits points, recolle les bords
  6. plus grande composante connexe               -> la main
  7. boîte englobante carrée, niveaux de gris, étirement d'histogramme, 28 x 28
"""
import cv2
import numpy as np

import config


class SeuilsPeau:
    """Intervalles de Cb et Cr considérés comme de la peau."""

    def __init__(self, cb_min=config.CB_MIN, cb_max=config.CB_MAX,
                 cr_min=config.CR_MIN, cr_max=config.CR_MAX):
        self.cb_min, self.cb_max = cb_min, cb_max
        self.cr_min, self.cr_max = cr_min, cr_max

    def __repr__(self):
        return (f"Cb dans [{self.cb_min}, {self.cb_max}], "
                f"Cr dans [{self.cr_min}, {self.cr_max}]")


def extraire_zone(image, zone=config.ZONE):
    """Découpe la zone où l'on signe. Renvoie la zone et la position (x1, y1, x2, y2)."""
    h, w = image.shape[:2]
    x1, y1 = int(zone[0] * w), int(zone[1] * h)
    x2, y2 = int(zone[2] * w), int(zone[3] * h)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def masque_peau(image_bgr, seuils):
    """Masque binaire (0 ou 255) des pixels dont Cb et Cr sont dans les intervalles de peau.

    Attention : OpenCV range les composantes dans l'ordre Y, Cr, Cb.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    bas = np.array([0, seuils.cr_min, seuils.cb_min], dtype=np.uint8)
    haut = np.array([255, seuils.cr_max, seuils.cb_max], dtype=np.uint8)
    return cv2.inRange(ycrcb, bas, haut)


def calibrer_peau(image_bgr, k=2.5, marge=6):
    """Apprend les seuils de peau sur un carré au centre de l'image, où l'utilisateur
    a posé sa paume. On prend moyenne +/- k écarts-types pour Cb et Cr."""
    h, w = image_bgr.shape[:2]
    c = min(h, w) // 6
    patch = image_bgr[h // 2 - c:h // 2 + c, w // 2 - c:w // 2 + c]
    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb).reshape(-1, 3).astype(float)
    cr, cb = ycrcb[:, 1], ycrcb[:, 2]

    def intervalle(v):
        bas = int(max(0, v.mean() - k * v.std() - marge))
        haut = int(min(255, v.mean() + k * v.std() + marge))
        return bas, haut

    cb_min, cb_max = intervalle(cb)
    cr_min, cr_max = intervalle(cr)
    return SeuilsPeau(cb_min, cb_max, cr_min, cr_max)


def masque_mouvement(image_bgr, fond_bgr, seuil=None):
    """Pixels qui diffèrent du fond appris (image prise sans la main).
    Sans seuil donné, on utilise la méthode d'Otsu sur l'image de différence."""
    diff = cv2.absdiff(cv2.GaussianBlur(image_bgr, (5, 5), 0),
                       cv2.GaussianBlur(fond_bgr, (5, 5), 0))
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    if seuil is None:
        _, m = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Otsu donne un seuil trop bas quand il n'y a presque rien dans l'image
        if _ < 15:
            _, m = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    else:
        _, m = cv2.threshold(diff, seuil, 255, cv2.THRESH_BINARY)
    return m


def boucher_petits_trous(masque, aire_max_ratio=0.003):
    """Remplit les petits trous à l'intérieur de la main (reflets, ombres).
    Les grands trous sont gardés : le rond du « O » ou du « F » aide à reconnaître la lettre."""
    inverse = cv2.bitwise_not(masque)
    n, etiquettes, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=4)
    h, w = masque.shape
    rempli = masque.copy()
    for i in range(1, n):
        x, y, bw, bh, aire = stats[i]
        touche_bord = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touche_bord and aire < aire_max_ratio * masque.size:
            rempli[etiquettes == i] = 255
    return rempli


def plus_grande_composante(masque, aire_min_ratio=0.02):
    """Garde seulement la plus grande composante connexe, si elle est assez grande."""
    n, etiquettes, stats, _ = cv2.connectedComponentsWithStats(masque, connectivity=8)
    if n <= 1:
        return np.zeros_like(masque)
    aires = stats[1:, cv2.CC_STAT_AREA]
    i = 1 + int(np.argmax(aires))
    if stats[i, cv2.CC_STAT_AREA] < aire_min_ratio * masque.size:
        return np.zeros_like(masque)
    return np.where(etiquettes == i, 255, 0).astype(np.uint8)


def taille_impaire(x):
    x = max(3, int(round(x)))
    return x if x % 2 == 1 else x + 1


def nettoyer_masque(masque):
    """Filtre médian, ouverture, fermeture, petits trous bouchés, plus grande composante.
    La taille des filtres est proportionnelle à la taille de l'image (environ 1,5 % et 2,5 %)."""
    etapes = {}
    d = min(masque.shape)
    m = cv2.medianBlur(masque, taille_impaire(d * 0.02))
    etapes["median"] = m
    el_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (taille_impaire(d * 0.015),) * 2)
    el_f = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (taille_impaire(d * 0.025),) * 2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, el_o)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, el_f)
    etapes["morphologie"] = m
    m = boucher_petits_trous(m)
    m = plus_grande_composante(m)
    etapes["main"] = m
    return m, etapes


def boite_carree(masque, marge=0.15):
    """Boîte englobante carrée (x1, y1, x2, y2) autour du masque, avec une marge."""
    ys, xs = np.nonzero(masque)
    if len(xs) == 0:
        return None
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cote = max(x2 - x1, y2 - y1) * (1 + 2 * marge)
    h, w = masque.shape
    x1, x2 = int(max(0, cx - cote / 2)), int(min(w, cx + cote / 2))
    y1, y2 = int(max(0, cy - cote / 2)), int(min(h, cy + cote / 2))
    return x1, y1, x2, y2


def vers_cnn(image_bgr, boite, masque=None, masquer_fond=False, taille=config.TAILLE_CNN):
    """Découpe la main, passe en niveaux de gris, étire l'histogramme et réduit en 28 x 28.

    Renvoie un tableau float entre 0 et 1 de forme (28, 28), comme Sign MNIST.
    """
    x1, y1, x2, y2 = boite
    gris = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if masquer_fond and masque is not None:
        m = masque[y1:y2, x1:x2] > 0
        gris[~m] = gris[m].mean() if m.any() else 0
    # on remet le carré au bon format s'il touche le bord de l'image
    h, w = gris.shape
    if h != w:
        c = max(h, w)
        carre = np.full((c, c), gris.mean(), np.float32)
        carre[(c - h) // 2:(c - h) // 2 + h, (c - w) // 2:(c - w) // 2 + w] = gris
        gris = carre
    petit = cv2.resize(gris, (taille, taille), interpolation=cv2.INTER_AREA)
    # étirement linéaire de l'histogramme entre 0 et 1
    mn, mx = petit.min(), petit.max()
    return (petit - mn) / (mx - mn + 1e-6)


def chaine_complete(image_bgr, seuils, fond_bgr=None, zone=config.ZONE, masquer_fond=False):
    """Applique toute la chaîne à une image de la webcam.

    Renvoie un dictionnaire avec toutes les étapes intermédiaires. Les clés utiles :
      'entree_cnn' : image 28 x 28 prête pour le CNN (None si pas de main)
      'boite'      : boîte de la main dans l'image complète (None si pas de main)
    """
    r = {}
    roi, (zx1, zy1, _, _) = extraire_zone(image_bgr, zone)
    r["zone"] = roi
    r["position_zone"] = (zx1, zy1)
    m = masque_peau(roi, seuils)
    r["peau"] = m
    if fond_bgr is not None:
        fond_roi, _ = extraire_zone(fond_bgr, zone)
        mv = masque_mouvement(roi, fond_roi)
        r["mouvement"] = mv
        m = cv2.bitwise_and(m, mv)
        r["peau_et_mouvement"] = m
    propre, etapes = nettoyer_masque(m)
    r.update(etapes)
    boite = boite_carree(propre)
    if boite is None:
        r["boite"] = None
        r["entree_cnn"] = None
        return r
    r["entree_cnn"] = vers_cnn(roi, boite, propre, masquer_fond)
    bx1, by1, bx2, by2 = boite
    r["boite"] = (bx1 + zx1, by1 + zy1, bx2 + zx1, by2 + zy1)
    return r


def panneau_debug(r, hauteur=160):
    """Petite image qui montre les étapes (zone, peau, masque final, entrée du CNN)."""
    def redim(img):
        if img is None:
            img = np.zeros((hauteur, hauteur), np.uint8)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * hauteur / h), hauteur),
                          interpolation=cv2.INTER_NEAREST)

    morceaux = [redim(r.get("zone")), redim(r.get("peau")), redim(r.get("main")),
                redim(r.get("entree_cnn"))]
    return np.hstack(morceaux)
