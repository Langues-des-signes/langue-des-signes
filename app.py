"""Application tout-en-un : enregistrer des signes, traduire en direct, entraîner, comparer.

Lancement : double-clic sur lancer.bat (ou python app.py).

  Menu      : choisir le mode (boutons ou touches 1, 2, 3). Chaque lancement de l'appli
              est une « séance » anonyme : pas besoin de donner son nom.
  Apprendre : l'application montre une image d'exemple de chaque lettre, compte 3, 2, 1 et
              enregistre. Pour J et Z (lettres en mouvement), elle enregistre 15 gestes.
  Direct    : reconnaissance en direct ; si la lettre est fausse, on garde le signe et on
              tape la bonne lettre au clavier : l'application l'apprend tout de suite.
  Entraîner : réentraîne les modèles, affiche les résultats, supprime des données.
  Comparer  : test des 3 moteurs sur les mêmes images de la webcam (scores pour le rapport).
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
import pretraitement as pre
import references
from outils import (compter_exemples, enregistrer_exemple, enregistrer_sequence, supprimer,
                    tout_supprimer, matrice_confusion)
from reconnaissance import Reconnaisseur, Lissage

FENETRE = "Langue des signes"
CAM_L, CAM_H = 640, 480          # taille de l'image de la webcam dans la fenêtre
PANNEAU = 400                    # largeur du panneau de droite
BARRE = 60                       # hauteur de la barre du bas
LARGEUR, HAUTEUR = CAM_L + PANNEAU, CAM_H + BARRE

# Ordre de « Apprendre » : les 26 lettres puis « autre mouvement » (pour J et Z)
ORDRE = config.ALPHABET + [config.AUTRE]
GESTES = config.LETTRES_MOUVEMENT + [config.AUTRE]


BLANC, GRIS, NOIR = (255, 255, 255), (170, 170, 170), (0, 0, 0)
VERT, ROUGE, ORANGE, BLEU = (90, 200, 90), (230, 80, 80), (250, 170, 60), (90, 150, 250)
FOND = (32, 32, 38)

# Rappel écrit, en plus de l'image d'exemple
DESCRIPTIONS = {
    "A": "Poing fermé, pouce collé sur le côté de l'index.",
    "B": "Main à plat, doigts collés vers le haut, pouce replié.",
    "C": "Main arrondie en forme de C.",
    "D": "Index levé, les autres doigts font un rond avec le pouce.",
    "E": "Doigts repliés sur le pouce, ongles visibles.",
    "F": "Pouce et index font un rond, les 3 autres doigts levés.",
    "G": "Index et pouce tendus à l'horizontale, parallèles.",
    "H": "Index et majeur tendus à l'horizontale, collés.",
    "I": "Petit doigt levé, poing fermé.",
    "J": "GESTE : petit doigt levé (comme I), dessine un J en l'air.",
    "K": "Index et majeur levés en V, pouce posé entre les deux.",
    "L": "Pouce et index forment un L.",
    "M": "Pouce caché sous 3 doigts repliés.",
    "N": "Pouce caché sous 2 doigts repliés.",
    "O": "Tous les doigts arrondis touchent le pouce : un O.",
    "P": "Comme K, mais la main tournée vers le bas.",
    "Q": "Comme G, mais pointé vers le bas.",
    "R": "Index et majeur levés et croisés.",
    "S": "Poing fermé, pouce devant les doigts.",
    "T": "Poing fermé, pouce glissé entre index et majeur.",
    "U": "Index et majeur levés et collés.",
    "V": "Index et majeur levés et écartés.",
    "W": "Index, majeur et annulaire levés et écartés.",
    "X": "Index levé et plié en crochet.",
    "Y": "Pouce et petit doigt tendus.",
    "Z": "GESTE : index tendu, dessine un Z en l'air.",
    config.AUTRE: "Bouge la main n'importe comment SANS faire J ni Z : passe d'une "
                  "lettre à l'autre, tourne la main, approche-la... Ça apprend au modèle "
                  "à ne pas voir des J et des Z partout.",
}


# Les 3 moteurs de reconnaissance, en mots simples : (nom court, explication)
MOTEURS = {
    "points": ("Squelette de la main",
               "MediaPipe place 21 points sur la main ; la lettre est reconnue d'après leur "
               "position. C'est le moteur le plus fiable : à utiliser par défaut."),
    "cnn-mp": ("Photo de la main + CNN",
               "MediaPipe trouve la main, puis un réseau de neurones (CNN) regarde une "
               "petite photo 28 x 28 de la main."),
    "cnn-peau": ("Couleur de peau + CNN",
                 "Traitement d'image classique (couleur de peau en YCbCr, filtre "
                 "médian, morphologie) pour trouver la main, puis le CNN. Mets ta main dans le "
                 "cadre jaune."),
}
ORDRE_MOTEURS = ["points", "cnn-mp", "cnn-peau"]
COTE = {"gauche": "GAUCHE", "droite": "DROITE", "les deux": "gauche ou droite"}[
    config.MAIN_UTILISEE]

# Message affiché en arrivant sur chaque écran : ce qu'on y fait
EXPLICATIONS_ECRAN = {
    "apprendre": f"Regarde le dessin de la lettre, fais pareil avec ta main {COTE}, puis "
                 "appuie sur espace : l'appli enregistre ta main pendant quelques secondes.",
    "direct": f"Fais une lettre avec ta main {COTE} et garde-la une seconde : elle s'écrit en "
              "bas. Si c'est faux, garde le signe et tape la bonne lettre au clavier.",
    "entrainer": "Clique sur « Mettre à jour la reconnaissance » pour que l'appli apprenne "
                 "tous les signes enregistrés. Le score s'affiche à gauche.",
    "comparer": "Test des 3 moteurs : pour chaque lettre, fais le signe DANS LE CADRE JAUNE et "
                "appuie sur espace. Les 3 moteurs regardent les mêmes images ; les scores "
                "sont enregistrés à la fin dans le dossier des résultats.",
}

# Dossiers qu'on peut ouvrir depuis l'appli
DOSSIERS = {"dossier_figures": config.DOSSIER_FIGURES,
            "dossier_donnees": config.DOSSIER_DONNEES}


def ouvrir_dossier(chemin):
    """Ouvre un dossier dans l'explorateur de fichiers."""
    chemin.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(str(chemin))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(chemin)])
    else:
        subprocess.Popen(["xdg-open", str(chemin)])


# ---------------------------------------------------------------- texte avec accents
_polices = {}


def police(taille):
    """Police TrueType (cv2.putText ne sait pas écrire les accents)."""
    if taille not in _polices:
        for nom in ["arial.ttf", "C:/Windows/Fonts/arial.ttf", "segoeui.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"]:
            try:
                _polices[taille] = ImageFont.truetype(nom, taille)
                break
            except OSError:
                pass
        else:
            _polices[taille] = ImageFont.load_default()
    return _polices[taille]


def couper_lignes(texte, taille, largeur):
    """Coupe un texte en lignes qui tiennent dans la largeur donnée (en pixels)."""
    f = police(taille)
    lignes = []
    for paragraphe in texte.split("\n"):
        ligne = ""
        for mot in paragraphe.split(" "):
            essai = (ligne + " " + mot).strip()
            if f.getlength(essai) <= largeur or not ligne:
                ligne = essai
            else:
                lignes.append(ligne)
                ligne = mot
        lignes.append(ligne)
    return lignes


class Tache:
    """Entraînement lancé en arrière-plan pour que la fenêtre ne se fige pas."""

    def __init__(self, nom, fonction, a_la_fin=None):
        self.nom = nom
        self.lignes = []
        self.finie = False
        self.resultat = None
        self.debut = time.time()

        def lancer():
            try:
                self.resultat = fonction(self.log)
            except Exception as e:   # on affiche l'erreur au lieu de planter
                self.log(f"ERREUR : {e}")
            if a_la_fin is not None:
                try:
                    a_la_fin(self)
                except Exception as e:
                    self.log(f"ERREUR : {e}")
            self.finie = True

        self.fil = threading.Thread(target=lancer, daemon=True)
        self.fil.start()

    def log(self, texte):
        print(texte)
        self.lignes.append(str(texte))


# ---------------------------------------------------------------- application
class App:
    def __init__(self, source=0, miroir=True, fps_video=None):
        self.source = source
        self.miroir = miroir
        self.fps_video = fps_video          # pour les tests sur une vidéo
        references.telecharger_en_fond()     # images d'exemple (une seule fois)
        self.rec = Reconnaisseur(avec_mediapipe=True)
        self.lissage = Lissage()
        self.etat = "accueil"
        self.methode = None
        # Chaque lancement = une séance anonyme. Elle sert seulement à remplacer une lettre
        # refaite pendant la même séance et à mesurer un score sur une séance jamais vue.
        self.seance = time.strftime("%Y%m%d%H%M%S")
        self.personne = "s" + self.seance
        self.numero = 0
        self.tache = None
        self.test_seance = False            # score mesuré sur la dernière séance ?
        self.message, self.fin_message = "", 0
        self.comptes = compter_exemples()
        self.boutons = []
        self.clic = None
        self.continuer = True
        self.debug = False                  # étapes du traitement affichées
        self.avance = False                 # options avancées du direct
        self.survol = None
        self.alphabet = False               # grille des 26 lettres par-dessus la webcam
        self.plein_ecran = False
        self.confirmer_suppression = False
        self.confirmer_tout = False          # « Tout effacer » demande 2 clics
        self.t = 0.0                        # temps de l'image en cours (secondes)
        self.derniere_image = np.zeros((CAM_H, CAM_L, 3), np.uint8)
        self.pour_capture = None
        self.derniere_vue = self.derniere_image
        # Apprendre
        self.i_lettre = 0
        self.phase = "pret"
        self.debut_phase = 0.0
        self.enregistrees = 0
        self.repetition = 0
        self.geste = []
        # Direct
        self.correction = None              # lettre en cours d'enregistrement
        self.corrigees = 0
        self.tampon = deque()               # (temps, main) des dernières secondes (J et Z)
        self.geste_vu = None                # (lettre, temps de fin d'affichage)
        self.pause_geste = 0.0
        # Comparer les moteurs
        self.cmp = None
        self.duree_detection = 0.0          # temps de MediaPipe pour l'image en cours

    # -------------------------------------------------------------- outils
    def dire(self, texte, duree=5.0):
        """Message bien visible en haut de l'image (explique ce qui vient de se passer)."""
        self.message, self.fin_message = texte, time.time() + duree

    def nb(self, lettre, personne=None):
        """Nombre d'exemples de la lettre dans cette séance (ou dans une autre)."""
        return self.comptes.get(personne or self.personne, {}).get(lettre, 0)

    def total(self, lettre):
        """Nombre d'exemples de la lettre, toutes séances confondues."""
        return sum(d.get(lettre, 0) for d in self.comptes.values())

    def ajouter_compte(self, lettre):
        d = self.comptes.setdefault(self.personne, {})
        d[lettre] = d.get(lettre, 0) + 1

    def enregistrer(self, image, main, lettre):
        if main is None:
            return False
        ok = enregistrer_exemple(image, main, lettre, self.personne, self.numero, self.seance)
        if ok:
            self.numero += 1
            self.ajouter_compte(lettre)
        return ok

    def lancer_tache(self, nom, fonction, a_la_fin=None):
        if self.tache is not None and not self.tache.finie:
            self.dire("Un entraînement est déjà en cours, patience...")
            return False
        self.tache = Tache(nom, fonction, a_la_fin)
        return True

    def seance_de_test(self):
        """La séance complète la plus récente (pour le score « jamais vue »). Une séance de
        quelques corrections en direct n'est pas un bon test : on prend la dernière séance
        où au moins 20 lettres fixes ont été enregistrées (sinon la plus récente)."""
        if not self.test_seance or len(self.comptes) < 2:
            return None
        completes = [s for s, d in self.comptes.items()
                     if sum(1 for l in config.LETTRES if d.get(l, 0) > 0) >= 20]
        return max(completes) if completes else max(self.comptes)

    def entrainer_points(self, rapide=False):
        import entrainer_points
        test = None if rapide else self.seance_de_test()

        def fonction(log):
            r = entrainer_points.entrainer(test, rapide=rapide, log=log)
            if not rapide:
                entrainer_points.entrainer_gestes(test, log=log)
            return r

        def fin(t):
            self.rec.charger_modeles(cnn=False)
            if self.methode is None or self.methode not in self.rec.disponibles():
                self.methode = "points" if "points" in self.rec.disponibles() else None
            t.log("Terminé : la nouvelle reconnaissance est utilisée tout de suite.")

        return self.lancer_tache("Mise à jour de la reconnaissance" if not rapide
                                 else "Apprentissage de la correction", fonction, fin)

    def entrainer_cnn(self):
        test = self.seance_de_test()

        def fonction(log):
            log("Chargement de TensorFlow...")
            import entrainer_cnn
            return entrainer_cnn.affiner(test, epoques=40, log=log)

        def fin(t):
            if t.resultat is not None:
                self.rec.charger_modeles(points=False, gestes=False)
                t.log("Terminé : le CNN amélioré est utilisé tout de suite.")

        return self.lancer_tache("Amélioration du CNN", fonction, fin)

    def aller(self, etat):
        self.etat = etat
        self.comptes = compter_exemples()
        self.alphabet = False
        self.confirmer_suppression = False
        if etat in EXPLICATIONS_ECRAN:
            self.dire(EXPLICATIONS_ECRAN[etat], 7)
        if etat == "direct":
            dispo = self.rec.disponibles()
            if self.methode not in dispo:
                self.methode = "points" if "points" in dispo else (dispo[0] if dispo else None)
            self.lissage = Lissage()
            self.correction = None
            self.tampon.clear()
        if etat == "apprendre":
            self.phase = "pret"
            # on commence à la première lettre pas encore faite pendant cette séance
            self.i_lettre = 0
            for i, l in enumerate(ORDRE):
                if self.nb(l) < self.objectif(l):
                    self.i_lettre = i
                    break

    def objectif(self, lettre):
        return config.NB_REPETITIONS if lettre in GESTES else config.NB_PAR_LETTRE

    def basculer_plein_ecran(self):
        self.plein_ecran = not self.plein_ecran
        try:
            cv2.setWindowProperty(FENETRE, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.plein_ecran
                                  else cv2.WINDOW_NORMAL)
            if not self.plein_ecran:
                cv2.resizeWindow(FENETRE, LARGEUR, HAUTEUR)
        except cv2.error:
            pass
        if self.plein_ecran:
            self.dire("Plein écran. Appuie sur F11 ou sur le bouton « Plein écran » du menu "
                      "pour revenir en fenêtre.", 4)

    # -------------------------------------------------------------- dessin
    def nouveau_canevas(self):
        self.canevas = np.zeros((HAUTEUR, LARGEUR, 3), np.uint8)
        self.canevas[:] = FOND[::-1]
        self.textes = []
        self.boutons = []

    def texte(self, x, y, texte, taille=20, couleur=BLANC, centre=False, largeur=None):
        """Ajoute un texte (dessiné à la fin avec PIL). Renvoie le y sous le texte."""
        lignes = couper_lignes(texte, taille, largeur) if largeur else [texte]
        for ligne in lignes:
            self.textes.append((x, y, ligne, taille, couleur, centre))
            y += int(taille * 1.3)
        return y

    def rectangle(self, x1, y1, x2, y2, couleur, plein=True):
        cv2.rectangle(self.canevas, (x1, y1), (x2, y2), couleur[::-1], -1 if plein else 2)

    def bouton(self, x, y, l, h, etiquette, action, couleur=(60, 60, 75), sous_titre=None,
               actif=True):
        """Bouton cliquable. Avec un sous-titre, le titre est en haut et une phrase
        d'explication en dessous. Un bouton inactif est grisé (mais reste cliquable pour
        expliquer pourquoi il est inactif)."""
        if not actif:
            couleur = (45, 45, 50)
        survole = self.survol == action
        self.rectangle(x, y, x + l, y + h, tuple(min(255, c + 25) for c in couleur)
                       if survole else couleur)
        self.rectangle(x, y, x + l, y + h, (200, 200, 220) if survole else (110, 110, 130),
                       plein=False)
        coul_texte = BLANC if actif else (130, 130, 130)
        if sous_titre:
            self.texte(x + 12, y + 7, etiquette, 19, coul_texte)
            self.texte(x + 12, y + 32, sous_titre, 13, (215, 215, 225), largeur=l - 24)
        else:
            taille = 17          # on réduit le texte s'il ne tient pas dans le bouton
            while taille > 11 and police(taille).getlength(etiquette) > l - 12:
                taille -= 1
            self.texte(x + l // 2, y + h // 2 - taille // 2 - 2, etiquette, taille,
                       coul_texte, centre=True)
        self.boutons.append(((x, y, x + l, y + h), action))

    def petit_bouton(self, x, y, l, h, etiquette, action):
        """Bouton discret (juste un cadre fin et un petit texte gris)."""
        survole = self.survol == action
        self.rectangle(x, y, x + l, y + h, (90, 90, 100) if survole else (60, 60, 68),
                       plein=False)
        self.texte(x + l // 2, y + 4, etiquette, 12, BLANC if survole else GRIS, centre=True)
        self.boutons.append(((x, y, x + l, y + h), action))

    def exemple(self, lettre, x, y, l, h):
        """Image d'exemple de la lettre ; à défaut, la lettre écrite en grand."""
        if lettre == config.AUTRE or not references.coller(self.canevas, lettre, x, y, l, h):
            self.rectangle(x, y, x + l, y + h, (50, 50, 60))
            self.texte(x + l // 2, y + h // 2 - 40, lettre if lettre != config.AUTRE
                       else "?", 60, GRIS, centre=True)

    def rendu(self):
        img = Image.fromarray(cv2.cvtColor(self.canevas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for x, y, t, taille, couleur, centre in self.textes:
            f = police(taille)
            if centre:
                x -= int(f.getlength(t) / 2)
            d.text((x, y), t, font=f, fill=couleur)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def coller_camera(self, image):
        self.canevas[:CAM_H, :CAM_L] = cv2.resize(image, (CAM_L, CAM_H))
        if self.alphabet:
            self.grille_alphabet()

    def grille_alphabet(self):
        """Les 26 images d'exemple par-dessus la webcam (bouton « Voir l'alphabet »)."""
        self.canevas[:CAM_H, :CAM_L] = (self.canevas[:CAM_H, :CAM_L] * 0.25).astype(np.uint8)
        cl, ch = CAM_L // 7, (CAM_H - 20) // 4
        for i, lettre in enumerate(config.ALPHABET):
            x, y = (i % 7) * cl, (i // 7) * ch
            self.exemple(lettre, x + 3, y + 3, cl - 6, ch - 26)
            self.texte(x + cl // 2, y + ch - 23, lettre, 18, BLANC, centre=True)
        self.texte(CAM_L - 10 - int(police(12).getlength(references.CREDIT)), CAM_H - 17,
                   references.CREDIT, 12, GRIS)

    def barre(self, texte, couleur=BLANC):
        self.rectangle(0, CAM_H, CAM_L, HAUTEUR, (10, 10, 12))
        taille = 26
        while taille > 14 and police(taille).getlength(texte) > CAM_L - 24:
            taille -= 2
        self.texte(12, CAM_H + (BARRE - taille) // 2 - 2, texte, taille, couleur)

    def dessiner_main(self, image, main):
        if main is not None:
            self.rec.mains.dessiner(image, main)

    def message_main(self, main):
        """Texte d'aide quand on ne voit pas la bonne main, sinon None."""
        if main is not None:
            return None
        if self.rec.detecteur.mauvaise_main:
            return f"Mauvaise main : signe avec ta main {COTE} !"
        return "Je ne vois pas ta main !"

    # -------------------------------------------------------------- une image
    def etape(self, image, temps):
        """Traite une image de la webcam et renvoie l'image complète de la fenêtre."""
        if image.shape[1] != CAM_L or image.shape[0] != CAM_H:
            image = cv2.resize(image, (CAM_L, CAM_H))
        self.t = temps
        self.derniere_image = image
        self.nouveau_canevas()
        main = None
        if self.etat != "entrainer":
            debut = time.perf_counter()
            main = self.rec.detecteur.detecter(image, temps * 1000)
            self.duree_detection = time.perf_counter() - debut
        getattr(self, "ecran_" + self.etat)(image, main)
        if self.etat != "accueil":
            self.texte(CAM_L + 20, HAUTEUR - 24, f"Main utilisée : {COTE.lower()}", 13, GRIS)
        self.pour_capture = (self.canevas.copy(), list(self.textes))   # rendu seulement si capture
        # petit bouton discret, toujours là : enregistre l'écran (sans lui) dans figures/
        self.petit_bouton(LARGEUR - 78, HAUTEUR - 27, 70, 22, "Capture", "capture")
        if self.message and time.time() < self.fin_message and self.etat != "entrainer":
            lignes = couper_lignes(self.message, 18, CAM_L - 40)[:4]
            h = 14 + 24 * len(lignes)
            self.rectangle(0, 0, CAM_L, h, (40, 40, 15))
            self.rectangle(0, 0, CAM_L, h, ORANGE[::-1], plein=False)
            for k, ligne in enumerate(lignes):
                self.texte(18, 8 + 24 * k, ligne, 18, (255, 230, 170))
        return self.rendu()

    def image_capture(self):
        """Image de la fenêtre telle qu'avant le bandeau de message et le bouton Capture."""
        canevas, textes = self.pour_capture
        sauve = self.canevas, self.textes
        self.canevas, self.textes = canevas.copy(), textes
        image = self.rendu()
        self.canevas, self.textes = sauve
        return image

    # -------------------------------------------------------------- accueil
    def ecran_accueil(self, image, main):
        vue = image.copy()
        self.dessiner_main(vue, main)
        self.coller_camera(vue)
        self.barre(self.message_main(main) or
                   "Ta main est bien vue (squelette vert) : la caméra marche.",
                   GRIS if main is None else VERT)
        x, l = CAM_L + 20, PANNEAU - 40
        self.texte(x, 12, "Langue des signes", 28)
        self.texte(x, 46, f"Alphabet américain (ASL) - main {COTE.lower()}", 15, GRIS)
        nb_ex = sum(sum(d.values()) for d in self.comptes.values())
        self.bouton(x, 74, l, 74, "1. Enregistrer des signes", "apprendre", (40, 90, 60),
                    "L'appli montre chaque lettre, on l'imite : elle apprend.")
        if nb_ex == 0:          # premier lancement : on montre par où commencer
            self.rectangle(x - 3, 71, x + l + 3, 151, (230, 170, 40), plein=False)
            self.rectangle(x + l - 112, 140, x + l - 4, 160, (230, 170, 40))
            self.texte(x + l - 58, 141, "Commence ici !", 14, NOIR, centre=True)
        self.bouton(x, 156, l, 74, "2. Traduire en direct", "direct", (40, 70, 110),
                    "Fais des lettres devant la caméra : l'appli écrit le mot.")
        self.bouton(x, 238, l, 74, "3. Mettre à jour et résultats", "entrainer", (100, 70, 40),
                    "Fait apprendre les signes enregistrés et donne le score.")
        self.bouton(x, 322, l // 2 - 5, 38, "Voir l'alphabet", "alphabet")
        self.bouton(x + l // 2 + 5, 322, l // 2 - 5, 38,
                    "Quitter le plein écran" if self.plein_ecran else "Plein écran (F11)",
                    "plein_ecran")
        completes = sum(1 for l2 in ORDRE if self.total(l2) >= self.objectif(l2))
        y = self.texte(x, 372, f"Signes enregistrés : {nb_ex} exemples, "
                               f"{completes}/{len(ORDRE)} lettres", 15, BLANC)
        y = self.texte(x, y, "Plus il y a de mains différentes (amis, famille...), mieux "
                             "l'appli reconnaît tout le monde.", 13, GRIS, largeur=l)
        self.bouton(x, max(y + 10, 452), l, 34, "Quitter", "quitter")

    # -------------------------------------------------------------- apprendre
    def ecran_apprendre(self, image, main):
        lettre = ORDRE[self.i_lettre]
        geste = lettre in GESTES
        vue = image.copy()
        self.dessiner_main(vue, main)
        if geste:
            self.apprendre_geste(vue, main, lettre)
        else:
            self.apprendre_statique(vue, image, main, lettre)
        self.coller_camera(vue)

        x, l = CAM_L + 20, PANNEAU - 40
        self.texte(x, 8, f"Lettre {self.i_lettre + 1} sur {len(ORDRE)}", 15, GRIS)
        self.texte(x + 65, 40, lettre if lettre != config.AUTRE else "~", 90, VERT,
                   centre=True)
        if lettre == config.AUTRE:
            self.texte(x + 65, 150, "Autre", 18, VERT, centre=True)
            self.texte(x + 65, 172, "mouvement", 18, VERT, centre=True)
        self.exemple(lettre, x + 135, 8, l - 135, 205)
        y = self.texte(x, 222, DESCRIPTIONS[lettre], 16, BLANC, largeur=l)
        unite = "gestes" if geste else "exemples"
        etat = f"Cette séance : {self.nb(lettre)} {unite} - en tout : {self.total(lettre)}"
        y = self.texte(x, y + 4, etat, 14, GRIS)
        y = max(y + 10, 318)
        if self.phase in ("pret", "fini"):
            self.bouton(x, y, l, 44, "Enregistrer cette lettre (espace)", "commencer",
                        (40, 90, 60))
        else:
            self.bouton(x, y, l, 44, "Arrêter (espace)", "arreter", (110, 50, 50))
        self.bouton(x, y + 52, l // 2 - 5, 36, "< Lettre d'avant", "precedente")
        self.bouton(x + l // 2 + 5, y + 52, l // 2 - 5, 36, "Passer >", "suivante")
        self.bouton(x, y + 96, l, 36, "J'ai fini : mettre à jour", "terminer", (100, 70, 40))
        self.bouton(x, y + 140, l // 2 - 5, 32, "Retour au menu", "menu")
        self.bouton(x + l // 2 + 5, y + 140, l // 2 - 5, 32,
                    "Cacher l'alphabet" if self.alphabet else "Voir l'alphabet", "alphabet")

    def demarrer_enregistrement(self):
        """Refaire une lettre pendant la même séance remplace la version précédente."""
        lettre = ORDRE[self.i_lettre]
        if self.nb(lettre) > 0:
            supprimer(self.personne, lettre)
            self.comptes = compter_exemples()
        self.alphabet = False
        self.phase, self.debut_phase = "decompte", self.t
        self.enregistrees, self.repetition, self.geste = 0, 0, []

    def lettre_suivante(self):
        self.dire(f"Lettre {ORDRE[self.i_lettre] if ORDRE[self.i_lettre] != config.AUTRE else '~'}"
                  " enregistrée ! On passe à la suivante.", 2)
        if self.i_lettre < len(ORDRE) - 1:
            self.i_lettre += 1
            self.phase = "pret"
        else:
            self.phase = "fini"

    def decompte(self, vue):
        reste = config.DECOMPTE - (self.t - self.debut_phase)
        if reste > 0:
            cv2.putText(vue, str(int(np.ceil(reste))), (CAM_L // 2 - 40, CAM_H // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 200, 255), 12)
        return reste <= 0

    def apprendre_statique(self, vue, image, main, lettre):
        if self.phase == "decompte" and self.decompte(vue):
            self.phase = "enregistre"
        if self.phase == "enregistre":
            if self.enregistrer(image, main, lettre):
                self.enregistrees += 1
            p = self.enregistrees / config.NB_PAR_LETTRE
            cv2.rectangle(vue, (0, CAM_H - 14), (int(CAM_L * p), CAM_H), (60, 60, 230), -1)
            cv2.circle(vue, (25, CAM_H - 40), 12, (0, 0, 255), -1)
            if self.enregistrees >= config.NB_PAR_LETTRE:
                self.lettre_suivante()
        probleme = self.message_main(main)
        if self.phase == "pret":
            self.barre(probleme or "1. Regarde le dessin   2. Fais pareil   3. Espace",
                       ROUGE if probleme else GRIS)
        elif self.phase == "decompte":
            self.barre(probleme or f"Prépare la lettre {lettre}...",
                       ROUGE if probleme else ORANGE)
        elif self.phase == "enregistre":
            self.barre(probleme or f"Enregistrement {self.enregistrees}/{config.NB_PAR_LETTRE}"
                       " : bouge un peu la main", ROUGE if probleme else BLANC)
        else:
            self.barre("Toutes les lettres sont faites, bravo ! « J'ai fini : mettre à jour ».",
                       VERT)

    def apprendre_geste(self, vue, main, lettre):
        """J, Z et « autre mouvement » : NB_REPETITIONS gestes de DUREE_GESTE secondes."""
        n = config.NB_REPETITIONS
        if self.phase == "decompte" and self.decompte(vue):
            self.phase, self.debut_phase, self.geste = "geste", self.t, []
        if self.phase == "pause" and self.t - self.debut_phase >= 1.0:
            self.phase, self.debut_phase, self.geste = "geste", self.t, []
        if self.phase == "geste":
            self.geste.append(main)
            p = min(1.0, (self.t - self.debut_phase) / config.DUREE_GESTE)
            cv2.rectangle(vue, (0, CAM_H - 14), (int(CAM_L * p), CAM_H), (60, 60, 230), -1)
            cv2.circle(vue, (25, CAM_H - 40), 12, (0, 0, 255), -1)
            if p >= 1.0:
                f = self.rec.mains.caracteristiques_sequence(self.geste)
                if f is not None:
                    enregistrer_sequence(f, lettre, self.personne)
                    self.ajouter_compte(lettre)
                    self.repetition += 1
                else:
                    self.dire("Main pas assez visible pendant le geste : on le refait.", 2)
                if self.repetition >= n:
                    self.lettre_suivante()
                else:
                    self.phase, self.debut_phase = "pause", self.t
        probleme = self.message_main(main) if self.phase != "pret" else None
        if self.phase == "pret":
            self.barre(f"Regarde le dessin : tu feras le geste {n} fois. Espace.", GRIS)
        elif self.phase == "decompte":
            self.barre("Prépare-toi, le premier geste arrive...", ORANGE)
        elif self.phase == "geste":
            self.barre(probleme or f"MAINTENANT ! Geste {self.repetition + 1}/{n}", ROUGE)
        elif self.phase == "pause":
            self.barre(f"Remets ta main en place... (geste {self.repetition + 1}/{n})",
                       ORANGE)
        else:
            self.barre("Toutes les lettres sont faites, bravo ! « J'ai fini : mettre à jour ».",
                       VERT)

    # -------------------------------------------------------------- direct
    def reconnaitre_geste(self, main):
        """Cherche un J ou un Z. On garde les dernières secondes de la main ; quand la main
        s'arrête après avoir bougé, on classe le mouvement complet UNE fois (J, Z ou
        « autre mouvement », par exemple le passage d'une lettre à une autre)."""
        if self.t < self.pause_geste or self.correction is not None:
            self.tampon.clear()     # juste après un geste : on repart de zéro
            return
        if main is None:
            return
        self.tampon.append((self.t, main))
        while self.tampon and self.t - self.tampon[0][0] > config.FENETRE_GESTE:
            self.tampon.popleft()
        if self.rec.gestes is None or len(self.tampon) < 10:
            return
        mains = [m for _, m in self.tampon]
        fin = [m for t, m in self.tampon if self.t - t <= 0.2]
        if len(fin) < 4 or self.rec.mains.amplitude_mouvement(fin) > config.SEUIL_IMMOBILE:
            return                  # la main bouge encore : on attend la fin du mouvement
        if self.rec.mains.amplitude_mouvement(mains) < config.SEUIL_MOUVEMENT:
            return                  # la main n'a pas vraiment bougé
        lettre, _, _ = self.rec.predire_geste(mains)
        self.tampon.clear()         # ce mouvement est traité, on n'y revient plus
        if lettre is not None:
            self.lissage.ajouter_geste(lettre, self.t)
            self.geste_vu = (lettre, self.t + 1.5)
            self.pause_geste = self.t + 0.5

    def vignettes_etapes(self, image, r, methode):
        """Images intermédiaires du moteur choisi, avec leur légende."""
        def recadrer(boite):
            x1, y1, x2, y2 = boite
            return image[y1:y2, x1:x2] if x2 - x1 > 10 and y2 - y1 > 10 else None

        if methode == "cnn-peau":
            et = r["etapes"] or {}
            return [(et.get("zone"), "1. Zone jaune"), (et.get("peau"), "2. Peau (YCbCr)"),
                    (et.get("main"), "3. Masque nettoyé"),
                    (et.get("entree_cnn"), "4. Entrée du CNN")]
        main = r["main"]
        if main is None:
            return [(None, "Pas de main")]
        morceau = recadrer(main["boite"])
        if methode == "cnn-mp":
            gris = None if morceau is None else cv2.cvtColor(morceau, cv2.COLOR_BGR2GRAY)
            x28 = (r["etapes"] or {}).get("entree_cnn")
            return [(morceau, "1. Main trouvée"), (gris, "2. Niveaux de gris"),
                    (x28, "3. Entrée du CNN")]
        # squelette : la main, les 21 points, puis les points normalisés (ce que voit le modèle)
        x1, y1, x2, y2 = main["boite"]
        avec_points = self.derniere_vue[y1:y2, x1:x2] if morceau is not None else None
        norm = np.full((120, 120, 3), 255, np.uint8)
        f = self.rec.mains.caracteristiques(main).reshape(21, 3)[:, :2]
        f = f - (f.min(0) + f.max(0)) / 2               # centrer le dessin
        pts = (f * 100 / (np.abs(f).max() * 2 + 1e-6) + 60).astype(int)
        for a, b in self.rec.mains.LIAISONS:
            cv2.line(norm, tuple(pts[a]), tuple(pts[b]), (0, 150, 0), 2)
        for p in pts:
            cv2.circle(norm, tuple(p), 3, (0, 0, 255), -1)
        return [(morceau, "1. Main trouvée"), (avec_points, "2. Les 21 points"),
                (norm, "3. Points normalisés")]

    def dessiner_etapes(self, vignettes):
        """Bande des étapes du traitement en bas de l'image de la webcam."""
        h, y0 = 110, CAM_H - 140
        self.rectangle(0, y0 - 4, CAM_L, CAM_H, (15, 15, 18))
        for k, (img, legende) in enumerate(vignettes):
            x0 = 8 + k * (h + 45)
            if img is not None:
                img = np.asarray(img)
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                ih, iw = img.shape[:2]
                kk = min(h / ih, (h + 30) / iw)
                img = cv2.resize(img, (max(1, int(iw * kk)), max(1, int(ih * kk))),
                                 interpolation=cv2.INTER_NEAREST)
                self.canevas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
            self.texte(x0, y0 + h + 4, legende, 13, BLANC)

    def ecran_direct(self, image, main):
        vue = image.copy()
        methode = self.methode
        r = {"probas": None, "boite": None, "main": main, "etapes": None}
        if methode is not None:
            r = self.rec.predire(image, methode, self.t * 1000, main=main)
        self.reconnaitre_geste(main)
        geste_actif = self.geste_vu is not None and self.t < self.geste_vu[1]
        lettre, _ = self.lissage.mettre_a_jour(
            r["probas"] if self.correction is None and not geste_actif else None, self.t)
        if geste_actif:
            lettre = self.geste_vu[0]

        # correction : on enregistre la bonne lettre puis on réentraîne
        if self.correction is not None:
            if self.enregistrer(image, main, self.correction):
                self.corrigees += 1
            if self.corrigees >= config.NB_CORRECTION:
                self.dire(f"{self.corrigees} exemples de {self.correction} ajoutés. "
                          "L'appli réapprend (une seconde)...", 4)
                self.correction = None
                self.entrainer_points(rapide=True)

        if methode == "cnn-peau":
            _, (x1, y1, x2, y2) = pre.extraire_zone(vue)
            cv2.rectangle(vue, (x1, y1), (x2, y2), (0, 255, 255), 2)
        self.dessiner_main(vue, main)
        self.derniere_vue = vue.copy()
        if r["boite"] is not None:
            x1, y1, x2, y2 = r["boite"]
            cv2.rectangle(vue, (x1, y1), (x2, y2), (255, 120, 0), 2)
        self.coller_camera(vue)
        if self.debug and methode is not None and not self.alphabet:
            self.dessiner_etapes(self.vignettes_etapes(image, r, methode))
        probleme = self.message_main(main) if methode != "cnn-peau" else None
        if probleme and not self.lissage.texte:
            self.barre(probleme, ROUGE)
        else:
            self.barre(self.lissage.texte[-34:] + "_")

        x, l = CAM_L + 20, PANNEAU - 40
        if methode is None:
            self.texte(x, 20, "Pas encore de reconnaissance.", 22, ORANGE)
            self.texte(x, 55, "Il faut d'abord enregistrer des signes : menu, « 1. Enregistrer "
                              "des signes », puis « J'ai fini : mettre à jour ».", 17, BLANC,
                       largeur=l)
            self.bouton(x, 200, l, 44, "Retour au menu", "menu")
            return
        self.texte(x, 8, f"Moteur : {MOTEURS[methode][0]}", 14, GRIS)
        if self.correction is not None:
            affichee, couleur = self.correction, ROUGE
        else:
            affichee, couleur = lettre, (VERT if lettre else GRIS)
        self.texte(x + 65, 35, affichee or "-", 100, couleur, centre=True)
        if affichee:
            self.exemple(affichee, x + 150, 28, l - 150, 140)
        if self.correction is not None:
            self.texte(x, 175, f"J'apprends {self.correction} : "
                               f"{self.corrigees}/{config.NB_CORRECTION}", 16, ROUGE)
        elif r["probas"] is not None:
            top = np.argsort(r["probas"])[::-1][:3]
            self.texte(x, 175, "Hésite entre : " + "  ".join(
                f"{config.LETTRES[k]} {100 * r['probas'][k]:.0f}%" for k in top), 15, BLANC)
        else:
            self.texte(x, 175, "Montre ta main à la caméra", 16, GRIS)

        if self.avance:
            self.panneau_avance(x, l, methode)
            return
        y = self.texte(x, 202, "Lettre fausse ? Garde le signe et tape la bonne lettre au "
                               "clavier : l'appli l'apprend. J et Z : fais le geste puis "
                               "arrête ta main.", 14, ORANGE, largeur=l)
        if self.tache is not None and (not self.tache.finie or
                                       time.time() - self.tache.debut < 15):
            dernier = self.tache.lignes[-1] if self.tache.lignes else "..."
            self.texte(x, y + 2, dernier, 13, BLEU, largeur=l)
        y = 318
        self.bouton(x, y, l // 2 - 5, 38,
                    "Cacher l'alphabet" if self.alphabet else "Voir l'alphabet", "alphabet")
        self.bouton(x + l // 2 + 5, y, l // 2 - 5, 38, "Effacer le texte", "effacer")
        self.bouton(x, y + 46, l, 38, "Options avancées (moteurs, étapes)", "avance")
        self.bouton(x, y + 92, l, 38, "Retour au menu", "menu")

    def panneau_avance(self, x, l, methode):
        """Options pour le rapport : choisir le moteur, voir les étapes, faire une capture."""
        dispo = self.rec.disponibles()
        self.texte(x, 200, "Moteur de reconnaissance (clique pour choisir) :", 13, GRIS)
        for k, m in enumerate(ORDRE_MOTEURS):
            nom = MOTEURS[m][0]
            if m == methode:
                nom = "► " + nom + " (utilisé)"
            elif m not in dispo:
                nom += " (pas encore prêt)"
            self.bouton(x, 220 + 38 * k, l, 32, nom, "moteur_" + m,
                        (40, 110, 60) if m == methode else (60, 60, 75), actif=m in dispo)
        y = self.texte(x, 338, MOTEURS[methode][1], 13, BLEU, largeur=l)
        y = max(y + 6, 396)
        self.bouton(x, y, l, 30, "Cacher les étapes" if self.debug else "Voir les étapes",
                    "debug")
        if methode == "cnn-peau":
            peau = "réglée sur ta main" if self.rec.seuils_calibres else "réglage de base"
            fond = "mémorisé" if self.rec.fond is not None else "aucun"
            self.texte(x, y + 34, f"Peau : {peau} - Décor : {fond}", 12, GRIS)
            self.bouton(x, y + 50, l // 3 - 4, 28, "Calibrer peau", "calibrer")
            self.bouton(x + l // 3 + 2, y + 50, l // 3 - 4, 28, "Mémoriser décor", "fond")
            self.bouton(x + 2 * (l // 3) + 4, y + 50, l // 3 - 4, 28, "Remettre à zéro",
                        "reinitialiser", (90, 50, 50))
            y += 50
        self.bouton(x, y + 36, l, 28, "Fermer les options", "avance")

    # -------------------------------------------------------------- entraîner
    def ecran_entrainer(self, image, main):
        # nombre d'exemples par lettre, toutes séances confondues
        self.texte(14, 8, f"Signes enregistrés ({len(self.comptes)} séance(s)) - "
                          "~ = autre mouvement", 16)
        for i, lettre in enumerate(ORDRE):
            c, lig = i // 14, i % 14
            x0, y = 14 + c * 320, 36 + 19 * lig
            n = self.total(lettre)
            coul = VERT if n >= self.objectif(lettre) else (ORANGE if n > 0 else ROUGE)
            self.texte(x0, y, "~" if lettre == config.AUTRE else lettre, 14, BLANC)
            self.texte(x0 + 30, y, f"{n}" + (" gestes" if lettre in GESTES else " exemples"),
                       14, coul)

        # journal de l'entraînement (avec, en haut, le message de ce qui vient de se passer)
        self.rectangle(8, 312, CAM_L - 8, CAM_H - 4, (20, 20, 24))
        y = 318
        if self.message and time.time() < self.fin_message:
            y = self.texte(18, y, self.message, 15, (255, 230, 170), largeur=CAM_L - 40) + 4
        elif self.tache is None:
            self.texte(18, y, "Clique sur « Mettre à jour la reconnaissance » à droite.",
                       16, GRIS)
        else:
            etat = "terminé" if self.tache.finie else "en cours..."
            y = self.texte(18, y, f"{self.tache.nom} : {etat}", 16, BLEU) + 2
        if self.tache is not None:
            # le score (ligne « RÉSULTAT ») reste toujours visible, puis les dernières lignes
            resultat = [l for l in self.tache.lignes if l.startswith("RÉSULTAT")]
            autres = [l for l in self.tache.lignes if not l.startswith("RÉSULTAT")]
            place = max(1, (CAM_H - 8 - y) // 18 - len(resultat))
            for k, ligne in enumerate(resultat + autres[-place:]):
                coul = VERT if ligne.startswith("RÉSULTAT") else BLANC
                self.texte(18, y + 18 * k, ligne[:85], 14 if k else 15, coul)
        self.barre("Le score « RÉSULTAT » s'affiche dans le cadre ci-dessus.", GRIS)

        x, l = CAM_L + 20, PANNEAU - 40
        self.texte(x, 10, "Mettre à jour et résultats", 22)
        self.bouton(x, 44, l, 66, "Mettre à jour la reconnaissance", "train_points",
                    (40, 90, 60), "Squelette de la main + gestes J et Z. Quelques secondes.")
        self.bouton(x, 118, l, 66, "Améliorer le CNN", "train_cnn", (40, 70, 110),
                    "Pour comparer les moteurs dans le rapport. Quelques minutes.")
        self.texte(x, 192, "Score mesuré sur :", 14, GRIS)
        choix = ("la dernière séance (jamais vue)" if self.test_seance
                 else "20 % des signes, au hasard")
        self.bouton(x, 212, l, 32, choix + "  (changer)", "test")
        self.bouton(x, 256, l // 2 - 5, 34, "Dossier des résultats", "dossier_figures")
        self.bouton(x + l // 2 + 5, 256, l // 2 - 5, 34, "Dossier des signes",
                    "dossier_donnees")
        if self.confirmer_suppression:
            self.bouton(x, 300, l, 34, "Sûr ? Clique encore pour effacer", "supprimer",
                        (150, 40, 40))
        else:
            self.bouton(x, 300, l, 34, "Effacer les signes de cette séance", "supprimer",
                        (90, 50, 50))
        if self.confirmer_tout:
            self.bouton(x, 344, l, 34, "Sûr ? Clique encore pour TOUT effacer", "tout_effacer",
                        (170, 30, 30))
        else:
            self.bouton(x, 344, l, 34, "Tout effacer (tous les signes)", "tout_effacer",
                        (90, 50, 50))
        self.bouton(x, 388, l, 34, "Comparer les 3 moteurs (test pour le rapport)",
                    "comparer", (40, 70, 110))
        self.bouton(x, 432, l, 34, "Retour au menu", "menu")

    # -------------------------------------------------------------- comparer les moteurs
    def ecran_comparer(self, image, main):
        """Chaque lettre est montrée ; pendant la mesure, les moteurs disponibles analysent
        exactement les mêmes images de la webcam, ce qui rend la comparaison juste."""
        c = self.cmp
        vue = image.copy()
        _, (x1, y1, x2, y2) = pre.extraire_zone(vue)
        cv2.rectangle(vue, (x1, y1), (x2, y2), (0, 255, 255), 2)
        self.dessiner_main(vue, main)
        lettre = config.LETTRES[c["i"]] if c["i"] < len(config.LETTRES) else None

        predictions = {}
        if lettre is not None and c["phase"] in ("pret", "decompte", "mesure"):
            for m in c["moteurs"]:
                debut = time.perf_counter()
                r = self.rec.predire(image, m, self.t * 1000, main=main)
                duree = time.perf_counter() - debut
                if m != "cnn-peau":                 # ces moteurs utilisent aussi MediaPipe
                    duree += self.duree_detection
                k = None if r["probas"] is None else int(np.argmax(r["probas"]))
                predictions[m] = k
                if c["phase"] == "mesure":
                    c["vrai"][m].append(config.LETTRES.index(lettre))
                    c["predit"][m].append(k)
                    c["ms"][m].append(1000 * duree)
        if c["phase"] == "decompte":
            reste = 2 - (self.t - c["debut"])
            if reste > 0:
                cv2.putText(vue, str(int(np.ceil(reste))), (CAM_L // 2 - 40, CAM_H // 2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 200, 255), 12)
            else:
                c["phase"] = "mesure"
        if c["phase"] == "mesure":
            c["n"] += 1
            p = c["n"] / config.NB_IMAGES_TEST
            cv2.rectangle(vue, (0, CAM_H - 14), (int(CAM_L * p), CAM_H), (60, 60, 230), -1)
            if c["n"] >= config.NB_IMAGES_TEST:
                self.cmp_suivante()
        self.coller_camera(vue)

        x, l = CAM_L + 20, PANNEAU - 40
        if c["phase"] == "fini":
            self.barre("Test terminé : scores enregistrés dans le dossier des résultats.", VERT)
            self.texte(x, 10, "Scores des 3 moteurs", 22)
            y = 48
            for m in c["moteurs"]:
                y = self.texte(x, y, f"{MOTEURS[m][0]} :", 16, BLANC)
                y = self.texte(x + 10, y, c["scores"][m].replace(" - ", "\n"), 15, VERT,
                               largeur=l - 10) + 6
            self.bouton(x, 360, l, 34, "Dossier des résultats", "dossier_figures")
            self.bouton(x, 404, l, 34, "Retour aux résultats", "entrainer")
            return
        self.texte(x, 8, f"Test des moteurs - lettre {c['i'] + 1} sur {len(config.LETTRES)}",
                   15, GRIS)
        self.texte(x + 65, 35, lettre, 90, VERT, centre=True)
        self.exemple(lettre, x + 150, 28, l - 150, 140)
        y = 178
        for m in c["moteurs"]:
            k = predictions.get(m)
            vu = "-" if k is None else config.LETTRES[k]
            coul = VERT if vu == lettre else (ROUGE if k is not None else GRIS)
            y = self.texte(x, y, f"{MOTEURS[m][0]} : {vu}", 15, coul)
        y = max(y + 10, 318)
        if c["phase"] == "pret":
            self.barre("Fais la lettre DANS LE CADRE JAUNE puis appuie sur espace.", GRIS)
            self.bouton(x, y, l, 40, "Mesurer cette lettre (espace)", "cmp_mesurer",
                        (40, 90, 60))
        else:
            self.barre("Garde le signe, bouge un tout petit peu la main...", ORANGE)
            self.bouton(x, y, l, 40, "Mesure en cours...", "rien", (70, 70, 80))
        self.bouton(x, y + 48, l, 34, "Passer cette lettre", "cmp_passer")
        self.bouton(x, y + 90, l, 34, "Arrêter et calculer les scores", "cmp_finir",
                    (100, 70, 40))

    def cmp_suivante(self):
        c = self.cmp
        c["i"] += 1
        c["phase"] = "pret"
        if c["i"] >= len(config.LETTRES):
            self.cmp_terminer()

    def cmp_terminer(self):
        """Calcule les scores, enregistre le tableau, le graphique et les matrices."""
        import csv
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        c = self.cmp
        c["phase"], c["scores"] = "fini", {}
        config.DOSSIER_FIGURES.mkdir(exist_ok=True)
        horodatage = time.strftime("%Y%m%d_%H%M%S")   # un jeu de fichiers par test
        lignes, noms, precisions = [], [], []
        for m in c["moteurs"]:
            vrai, predit = c["vrai"][m], c["predit"][m]
            if not vrai:
                c["scores"][m] = "aucune image mesurée"
                continue
            bons = sum(1 for v, p in zip(vrai, predit) if v == p)
            sans_main = sum(1 for p in predit if p is None)
            acc = 100 * bons / len(vrai)
            ms = float(np.mean(c["ms"][m]))
            c["scores"][m] = (f"{acc:.0f} % bien reconnues - main non trouvée "
                              f"{100 * sans_main / len(vrai):.0f} % - {ms:.0f} ms par image")
            lignes.append([MOTEURS[m][0], f"{acc:.1f}", f"{100 * sans_main / len(vrai):.1f}",
                           f"{ms:.1f}", len(vrai), len(set(vrai))])
            noms.append(MOTEURS[m][0])
            precisions.append(acc)
            avec = [(v, p) for v, p in zip(vrai, predit) if p is not None]
            if avec:
                matrice_confusion([v for v, _ in avec], [p for _, p in avec],
                                  f"Test - {MOTEURS[m][0]}",
                                  f"confusion_test_{m}_{horodatage}.png")
        # détail image par image : permet de regrouper plusieurs testeurs pour le rapport
        with open(config.DOSSIER_FIGURES / f"comparaison_details_{horodatage}.csv", "w",
                  newline="", encoding="utf-8-sig") as f:
            ecrivain = csv.writer(f, delimiter=";")
            ecrivain.writerow(["moteur", "lettre montrée", "lettre trouvée", "ms"])
            for m in c["moteurs"]:
                for v, p, ms in zip(c["vrai"][m], c["predit"][m], c["ms"][m]):
                    ecrivain.writerow([m, config.LETTRES[v],
                                       "" if p is None else config.LETTRES[p], f"{ms:.1f}"])
        with open(config.DOSSIER_FIGURES / f"comparaison_moteurs_{horodatage}.csv", "w",
                  newline="", encoding="utf-8-sig") as f:
            ecrivain = csv.writer(f, delimiter=";")
            ecrivain.writerow(["moteur", "précision (%)", "main non trouvée (%)",
                               "ms par image", "images", "lettres testées"])
            ecrivain.writerows(lignes)
        if noms:
            plt.figure(figsize=(7, 4))
            barres = plt.bar(noms, precisions, color=["#3b8f5a", "#3a6fa8", "#a8743a"][:len(noms)])
            for b, v in zip(barres, precisions):
                plt.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f} %", ha="center")
            plt.ylim(0, 105)
            plt.ylabel("lettres bien reconnues (%)")
            plt.title("Comparaison des moteurs sur les mêmes images")
            plt.tight_layout()
            plt.savefig(config.DOSSIER_FIGURES / f"comparaison_moteurs_{horodatage}.png",
                        dpi=130)
            plt.close()
        self.dire("Scores enregistrés dans le dossier des résultats (tableau "
                  "comparaison_moteurs, graphique et matrices de confusion).", 6)

    # -------------------------------------------------------------- actions
    def action(self, a):
        """Exécute un bouton et explique ce qui vient de se passer."""
        if a != "supprimer":
            self.confirmer_suppression = False
        if a != "tout_effacer":
            self.confirmer_tout = False
        if a == "quitter":
            self.continuer = False
        elif a in ("apprendre", "direct", "entrainer"):
            self.aller(a)
        elif a == "menu":
            if self.alphabet:
                self.alphabet = False
            elif self.etat == "direct" and self.correction is not None:
                self.correction = None
                self.dire("Correction annulée.", 2)
            else:
                self.aller("accueil")
        elif a == "alphabet":
            self.alphabet = not self.alphabet
            if self.alphabet:
                self.dire("Voici les 26 lettres. Clique sur « Cacher l'alphabet » (ou Échap) "
                          "pour revoir la caméra.", 4)
        elif a == "plein_ecran":
            self.basculer_plein_ecran()
        elif a == "avance":
            self.avance = not self.avance
            if self.avance:
                self.dire("Options avancées : choisis un moteur de reconnaissance, affiche "
                          "les étapes du traitement ou fais une capture pour le rapport.", 5)
            else:
                self.debug = False
        # apprendre
        elif a == "commencer":
            if self.phase == "fini":
                self.i_lettre = 0
            self.demarrer_enregistrement()
        elif a == "arreter":
            self.phase = "pret"
            self.dire("Enregistrement arrêté. Clique sur « Enregistrer cette lettre » pour "
                      "recommencer.", 3)
        elif a == "suivante":
            self.i_lettre, self.phase = (self.i_lettre + 1) % len(ORDRE), "pret"
        elif a == "precedente":
            self.i_lettre, self.phase = (self.i_lettre - 1) % len(ORDRE), "pret"
        elif a == "terminer":
            self.aller("entrainer")
            self.entrainer_points()
            self.dire("L'appli apprend les signes enregistrés. Le score s'affiche à gauche "
                      "dans quelques secondes.", 5)
        # direct
        elif a.startswith("moteur_"):
            m = a[len("moteur_"):]
            if m in self.rec.disponibles():
                self.methode = m
                self.lissage.votes.clear()
                self.dire(f"Moteur choisi : {MOTEURS[m][0]}. {MOTEURS[m][1]}", 6)
            elif m == "points":
                self.dire("Ce moteur n'est pas encore prêt : enregistre des signes puis "
                          "« Mettre à jour la reconnaissance ».", 5)
            else:
                self.dire("Ce moteur a besoin du fichier modeles/cnn.keras.", 5)
        elif a == "effacer":
            self.lissage.texte = ""
        elif a == "debug":
            self.debug = not self.debug
            if self.debug:
                self.dire("Les étapes du traitement sont affichées en bas de l'image : ce que "
                          f"fait le moteur « {MOTEURS[self.methode][0]} » pour trouver la "
                          "lettre.", 5)
        elif a == "capture":
            config.DOSSIER_FIGURES.mkdir(exist_ok=True)
            nom = f"capture_{self.etat}_{time.strftime('%Y%m%d_%H%M%S')}.png"
            if self.pour_capture is not None:
                cv2.imwrite(str(config.DOSSIER_FIGURES / nom), self.image_capture())
            self.dire(f"Capture enregistrée dans le dossier des résultats ({nom}).", 3)
        elif a == "calibrer":
            self.rec.seuils = pre.calibrer_peau(pre.extraire_zone(self.derniere_image)[0])
            self.rec.seuils_calibres = True
            self.dire("Couleur de peau apprise sur ce qu'il y avait AU CENTRE du cadre jaune. "
                      "Si ta paume n'y était pas, clique sur « Remettre à zéro ».", 6)
        elif a == "fond":
            self.rec.fond = self.derniere_image.copy()
            self.dire("Décor mémorisé : seul ce qui a changé depuis (ta main) est gardé. Si ta "
                      "main était dans le cadre jaune, clique sur « Remettre à zéro ».", 6)
        elif a == "reinitialiser":
            self.rec.seuils = pre.SeuilsPeau()
            self.rec.seuils_calibres = False
            self.rec.fond = None
            self.dire("Réglages remis à zéro : couleur de peau par défaut, aucun décor "
                      "mémorisé.", 4)
        # entraîner
        elif a == "train_points":
            if self.entrainer_points():
                self.dire("L'appli apprend les signes enregistrés. Le score s'affiche à "
                          "gauche dans quelques secondes.", 5)
        elif a == "train_cnn":
            if self.entrainer_cnn():
                self.dire("Le CNN réapprend sur vos photos de mains : quelques minutes. Tu "
                          "peux suivre l'avancement à gauche.", 5)
        elif a == "test":
            self.test_seance = not self.test_seance
            if self.test_seance:
                self.dire("Le score sera mesuré sur la dernière séance, que l'appli n'aura "
                          "pas vue pendant l'apprentissage : c'est le score honnête pour le "
                          "rapport (il faut au moins 2 séances).", 7)
            else:
                self.dire("Le score sera mesuré sur 20 % des signes tirés au hasard.", 4)
        elif a in DOSSIERS:
            try:
                ouvrir_dossier(DOSSIERS[a])
            except Exception as e:
                self.dire(f"Impossible d'ouvrir le dossier : {e}", 4)
        elif a == "comparer":
            if not self.rec.disponibles():
                self.dire("Aucun moteur n'est prêt.", 4)
            else:
                self.aller("comparer")
                self.cmp = {"i": 0, "phase": "pret", "debut": 0.0, "n": 0,
                            "moteurs": [m for m in ORDRE_MOTEURS
                                        if m in self.rec.disponibles()],
                            "vrai": {m: [] for m in ORDRE_MOTEURS},
                            "predit": {m: [] for m in ORDRE_MOTEURS},
                            "ms": {m: [] for m in ORDRE_MOTEURS}}
                if "points" not in self.cmp["moteurs"]:
                    self.dire("Le moteur « Squelette » n'est pas encore prêt : il ne sera pas "
                              "testé. Enregistre des signes puis « Mettre à jour » d'abord.", 7)
        elif a == "cmp_mesurer":
            self.cmp["phase"], self.cmp["debut"], self.cmp["n"] = "decompte", self.t, 0
        elif a == "cmp_passer":
            self.cmp_suivante()
        elif a == "cmp_finir":
            self.cmp_terminer()
        elif a == "tout_effacer":
            if not self.confirmer_tout:
                self.confirmer_tout = True
                self.dire("ATTENTION : tous les signes enregistrés (toutes les séances) et ce "
                          "que l'appli en a appris vont être effacés. Impossible de revenir "
                          "en arrière. Clique encore pour confirmer.", 8)
            else:
                self.confirmer_tout = False
                n = tout_supprimer()
                self.comptes = compter_exemples()
                self.rec.clf_points, self.rec.gestes = None, None
                self.rec.charger_modeles(points=False, gestes=False)   # CNN de base
                self.methode = None
                self.dire(f"Tout est effacé ({n} exemples). On repart de zéro : va dans "
                          "« 1. Enregistrer des signes ».", 6)
        elif a == "supprimer":
            if not self.confirmer_suppression:
                self.confirmer_suppression = True
                self.dire("Les signes enregistrés depuis le lancement de l'appli vont être "
                          "effacés. Clique encore pour confirmer.", 4)
            else:
                n = supprimer(self.personne)
                self.confirmer_suppression = False
                self.comptes = compter_exemples()
                if self.tache is None or self.tache.finie:
                    self.tache = Tache("Effacement", lambda log: log(
                        f"{n} exemples de cette séance effacés. Pense à mettre à jour."))

    def touche(self, k):
        if k < 0 or k == 255:
            return
        c = chr(k).upper() if 32 <= k < 127 else ""
        if k == 27:
            self.action("menu" if self.etat != "accueil" or self.alphabet else "quitter")
        elif self.etat == "accueil":
            self.action({"1": "apprendre", "2": "direct", "3": "entrainer"}.get(c, ""))
        elif self.etat == "apprendre":
            if k == 32:
                self.action("commencer" if self.phase in ("pret", "fini") else "arreter")
            elif k == 9:
                self.action("suivante")
            elif c in config.ALPHABET and self.phase in ("pret", "fini"):
                self.i_lettre, self.phase = ORDRE.index(c), "pret"
        elif self.etat == "direct":
            if c in config.LETTRES_MOUVEMENT:
                self.dire(f"{c} se fait avec un geste : on l'enregistre dans « 1. Enregistrer "
                          "des signes ».", 4)
            elif c in config.LETTRES and self.methode is not None:
                if self.nb(c) >= config.MAX_PAR_LETTRE:
                    self.dire(f"Déjà beaucoup d'exemples de {c} pendant cette séance. "
                              "Refais-la plutôt dans « Enregistrer des signes ».", 4)
                else:
                    self.correction, self.corrigees = c, 0
                    self.dire(f"Garde le signe {c} : j'enregistre quelques images puis je "
                              "l'apprends.", 3)
            elif k == 8:
                self.lissage.texte = self.lissage.texte[:-1]
            elif k == 32:
                self.lissage.texte += " "
        elif self.etat == "comparer" and self.cmp is not None:
            if k == 32 and self.cmp["phase"] == "pret":
                self.action("cmp_mesurer")
            elif k == 9 and self.cmp["phase"] == "pret":
                self.action("cmp_passer")
        elif self.etat == "entrainer":
            if c == "1":
                self.action("train_points")
            elif c == "2":
                self.action("train_cnn")

    def souris(self, evenement, x, y, _flags, _param):
        sous = None
        for (x1, y1, x2, y2), a in self.boutons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                sous = a
        self.survol = sous          # le bouton sous la souris est éclairci
        if evenement == cv2.EVENT_LBUTTONUP and sous is not None:
            self.clic = sous

    # -------------------------------------------------------------- boucle
    def ouvrir_camera(self):
        if isinstance(self.source, str):
            return cv2.VideoCapture(self.source)
        cam = cv2.VideoCapture(self.source, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") \
            else cv2.VideoCapture(self.source)
        if not cam.isOpened():
            cam = cv2.VideoCapture(self.source)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cam

    def lancer(self):
        cam = self.ouvrir_camera()
        # fenêtre redimensionnable (bouton agrandir de Windows, plein écran avec F11)
        cv2.namedWindow(FENETRE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(FENETRE, LARGEUR, HAUTEUR)
        cv2.setMouseCallback(FENETRE, self.souris)
        t0, n = time.time(), 0
        while self.continuer:
            ok, image = cam.read()
            if not ok:
                image = np.zeros((CAM_H, CAM_L, 3), np.uint8)
                cv2.putText(image, "Webcam introuvable", (120, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3)
                if isinstance(self.source, str):
                    break
                time.sleep(0.1)
            elif self.miroir:
                image = cv2.flip(image, 1)
            temps = n / self.fps_video if self.fps_video else time.time() - t0
            n += 1
            cv2.imshow(FENETRE, self.etape(image, temps))
            k = cv2.waitKeyEx(1)
            if k in (0x7A0000, 0xFFC8):         # F11 (Windows, Linux)
                self.basculer_plein_ecran()
            else:
                self.touche(k & 0xFF if k >= 0 else -1)
            if self.clic is not None:
                a, self.clic = self.clic, None
                self.action(a)
            try:
                if cv2.getWindowProperty(FENETRE, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        cam.release()
        cv2.destroyAllWindows()
        self.rec.fermer()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default=None, help="pour tester sur une vidéo")
    args = p.parse_args()
    source = args.video if args.video else args.camera
    App(source, miroir=args.video is None).lancer()


if __name__ == "__main__":
    main()
