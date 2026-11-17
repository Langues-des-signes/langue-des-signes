"""Regroupe les trois méthodes de reconnaissance, plus le lissage dans le temps.

  'cnn-peau' : chaîne classique (pretraitement.py) + CNN.  Aucune IA pour trouver la main.
  'cnn-mp'   : MediaPipe trouve la boîte de la main + CNN.
  'points'   : MediaPipe donne les 21 points + classifieur (entrainer_points.py).
"""
import time
from collections import Counter, deque

import numpy as np

import config
import pretraitement as pre
from outils import image_vers_28

METHODES = ["cnn-peau", "cnn-mp", "points"]


class Reconnaisseur:
    def __init__(self, avec_mediapipe=True):
        self.cnn = None
        self.clf_points = None
        self.gestes = None        # modèle des gestes J et Z
        self.detecteur = None
        self.seuils = pre.SeuilsPeau()
        self.seuils_calibres = False   # vrai après « Calibrer la peau »
        self.fond = None          # image du fond sans main, apprise avec la touche f
        self.masquer_fond = False
        self.charger_modeles()
        if avec_mediapipe:
            import mains
            self.mains = mains
            self.detecteur = mains.DetecteurMain(video=True)

    def charger_modeles(self, cnn=True, points=True, gestes=True):
        """(Re)charge les modèles depuis le dossier modeles/ (après un entraînement)."""
        if gestes:
            f = config.DOSSIER_MODELES / "gestes.joblib"
            if f.exists():
                import joblib
                self.gestes = joblib.load(f)
                print("Modèle des gestes chargé")
        if cnn:
            self.cnn = None
            for nom in ["cnn_affine.keras", "cnn.keras"]:
                f = config.DOSSIER_MODELES / nom
                if f.exists():
                    import keras
                    self.cnn = keras.models.load_model(f)
                    print("CNN chargé :", nom)
                    break
        if points:
            f = config.DOSSIER_MODELES / "points.joblib"
            if f.exists():
                import joblib
                self.clf_points = joblib.load(f)
                print("Classifieur des points chargé")

    def disponibles(self):
        d = []
        if self.cnn is not None:
            d.append("cnn-peau")
            if self.detecteur is not None:
                d.append("cnn-mp")
        if self.clf_points is not None and self.detecteur is not None:
            d.append("points")
        return d

    def fermer(self):
        if self.detecteur is not None:
            self.detecteur.fermer()

    def _cnn(self, x28):
        x = np.asarray(x28, np.float32)[None, :, :, None]
        return np.asarray(self.cnn(x, training=False))[0]

    def predire(self, image, methode, temps_ms, main="calculer"):
        """Renvoie un dictionnaire : 'probas' (24 valeurs ou None), 'boite', 'main', 'etapes'.
        On peut donner la main déjà détectée par MediaPipe (paramètre main)."""
        r = {"probas": None, "boite": None, "main": None, "etapes": None}
        if methode == "cnn-peau":
            et = pre.chaine_complete(image, self.seuils, self.fond,
                                     masquer_fond=self.masquer_fond)
            r["etapes"] = et
            if et["entree_cnn"] is not None:
                r["boite"] = et["boite"]
                r["probas"] = self._cnn(et["entree_cnn"])
            return r

        if isinstance(main, str):
            main = self.detecteur.detecter(image, temps_ms)
        r["main"] = main
        if main is None:
            return r
        r["boite"] = main["boite"]
        if methode == "points":
            f = self.mains.caracteristiques(main)[None, :]
            clf = self.clf_points   # copie locale : le modèle peut être remplacé pendant ce temps
            probas = np.zeros(len(config.LETTRES))
            probas[clf.classes_] = clf.predict_proba(f)[0]
            r["probas"] = probas
        else:  # cnn-mp
            x1, y1, x2, y2 = main["boite"]
            if x2 - x1 > 10 and y2 - y1 > 10:
                x28 = image_vers_28(image[y1:y2, x1:x2])
                r["etapes"] = {"entree_cnn": x28}
                r["probas"] = self._cnn(x28)
        return r


    def predire_geste(self, liste_mains):
        """Reconnaît J ou Z sur environ une seconde de mains.
        Renvoie (lettre ou None, probabilité, amplitude du mouvement)."""
        if self.gestes is None:
            return None, 0.0, 0.0
        amplitude = self.mains.amplitude_mouvement(liste_mains)
        if amplitude < config.SEUIL_MOUVEMENT:
            return None, 0.0, amplitude        # main immobile : pas de geste
        f = self.mains.caracteristiques_sequence(liste_mains)
        if f is None:
            return None, 0.0, amplitude
        modele, classes = self.gestes["modele"], self.gestes["classes"]
        probas = modele.predict_proba(f[None, :])[0]
        k = int(np.argmax(probas))
        lettre = classes[modele.classes_[k]]
        if lettre == config.AUTRE or probas[k] < config.CONFIANCE_GESTE:
            return None, float(probas[k]), amplitude
        return lettre, float(probas[k]), amplitude


class Lissage:
    """Vote majoritaire sur les dernières images, puis construction du mot.

    Une lettre est ajoutée au texte quand elle reste la même pendant DUREE_VALIDATION
    secondes ; un espace est ajouté quand la main disparaît pendant DUREE_ESPACE secondes.
    """

    def __init__(self):
        self.votes = deque(maxlen=config.NB_IMAGES_VOTE)
        self.texte = ""
        self.lettre_courante = None
        self.debut_lettre = time.time()
        self.derniere_main = time.time()
        self.deja_ajoutee = False
        self.dernier_ajout = -1e9   # moment où la dernière lettre a été ajoutée

    def ajouter_geste(self, lettre, maintenant):
        """Ajoute J ou Z. J commence par la forme du I : si un I vient d'être ajouté juste
        avant le geste, c'était le début du J, on le remplace."""
        if (lettre == "J" and self.texte.endswith("I")
                and maintenant - self.dernier_ajout < 3.0):
            self.texte = self.texte[:-1]
        self.texte += lettre
        self.votes.clear()
        self.dernier_ajout = maintenant

    def mettre_a_jour(self, probas, maintenant=None):
        maintenant = time.time() if maintenant is None else maintenant
        if probas is None:
            self.votes.append(None)
        else:
            self.derniere_main = maintenant
            k = int(np.argmax(probas))
            self.votes.append(k if probas[k] >= config.CONFIANCE_MIN else None)

        valides = [v for v in self.votes if v is not None]
        lettre, part = None, 0.0
        if valides:
            k, n = Counter(valides).most_common(1)[0]
            part = n / len(self.votes)
            if part >= 0.5:
                lettre = config.LETTRES[k]

        if lettre != self.lettre_courante:
            self.lettre_courante = lettre
            self.debut_lettre = maintenant
            self.deja_ajoutee = False
        elif (lettre is not None and not self.deja_ajoutee
              and maintenant - self.debut_lettre >= config.DUREE_VALIDATION):
            self.texte += lettre
            self.deja_ajoutee = True
            self.dernier_ajout = maintenant

        if (maintenant - self.derniere_main >= config.DUREE_ESPACE
                and self.texte and not self.texte.endswith(" ")):
            self.texte += " "
        return lettre, part
