"""Entraîne un classifieur sur les 63 caractéristiques des points de la main (MediaPipe).

On compare trois classifieurs : k plus proches voisins, SVM et
réseau de neurones (MLP). Le meilleur est enregistré dans modeles/points.joblib.

Utilisation :
    python entrainer_points.py                         # 20 % de test au hasard
    python entrainer_points.py --test-personne s20261110190000   # test sur une séance jamais vue
L'application (app.py) appelle directement la fonction entrainer().
"""
import argparse
import time

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from outils import (charger_points, charger_sequences, decouper, matrice_confusion,
                    il_y_a_des_points, CLASSES_GESTE)


def augmenter(X, y, n=3, bruit=0.02, graine=0):
    """Ajoute des copies légèrement bruitées et tournées des exemples d'entraînement."""
    rng = np.random.default_rng(graine)
    Xs, ys = [X], [y]
    for _ in range(n):
        pts = X.reshape(-1, 21, 3).copy()
        angle = rng.uniform(-0.25, 0.25, size=len(X))      # environ +/- 15 degrés
        c, s = np.cos(angle), np.sin(angle)
        x, yy = pts[:, :, 0].copy(), pts[:, :, 1].copy()
        pts[:, :, 0] = c[:, None] * x - s[:, None] * yy
        pts[:, :, 1] = s[:, None] * x + c[:, None] * yy
        pts += rng.normal(0, bruit, pts.shape)
        Xs.append(pts.reshape(len(X), -1).astype(np.float32))
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def classifieurs(rapide=False):
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(5))
    if rapide:   # instantané : utilisé après une correction en direct
        return {"kNN (k=5)": knn}
    return {
        "kNN (k=5)": knn,
        "SVM": make_pipeline(StandardScaler(),
                             CalibratedClassifierCV(SVC(C=10), ensemble=False)),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier((128, 64), max_iter=500, early_stopping=True,
                                           random_state=0)),
    }


def entrainer(test_personne=None, rapide=False, log=print):
    """Entraîne, enregistre le meilleur modèle et renvoie un dictionnaire de résultats.

    rapide=True : seulement le kNN (instantané), pour le réentraînement pendant le direct.
    """
    if not il_y_a_des_points():
        log("Aucune donnée : enregistrez d'abord des lettres.")
        return None
    X, y, pers = charger_points()
    lettres = [config.LETTRES[k] for k in np.unique(y)]
    log(f"{len(X)} exemples, {len(lettres)} lettres, {len(set(pers.tolist()))} séance(s)")
    if len(lettres) < 2:
        log("Il faut au moins 2 lettres différentes pour entraîner.")
        return None

    if test_personne is not None and test_personne not in set(pers.tolist()):
        log("Séance de test inconnue : test sur 20 % des signes au hasard.")
        test_personne = None
    a, t = decouper(pers, y, test_personne)
    if test_personne is not None and len(set(pers[a].tolist())) == 0:
        log("Il faut au moins 2 séances pour tester sur une séance jamais vue : "
            "test au hasard à la place.")
        test_personne = None
        a, t = decouper(pers, y)
    Xa, ya = augmenter(X[a], y[a])
    Xt, yt = X[t], y[t]

    meilleur, meilleure_acc, nom_meilleur, scores = None, -1, None, {}
    for nom, clf in classifieurs(rapide).items():
        debut = time.time()
        clf.fit(Xa, ya)
        acc = float((clf.predict(Xt) == yt).mean()) if len(yt) else float("nan")
        scores[nom] = acc
        log(f"{nom:10s} précision test = {100 * acc:.1f} %  ({time.time() - debut:.0f} s)")
        if meilleur is None or acc > meilleure_acc:
            meilleur, meilleure_acc, nom_meilleur = clf, acc, nom

    if not rapide:
        log(f"Meilleur : {nom_meilleur}")
        sur = ("une séance que l'appli n'a jamais vue" if test_personne
               else "20 % des signes tirés au hasard")
        log(f"RÉSULTAT : {100 * meilleure_acc:.0f} % des lettres bien reconnues ({sur}).")
        matrice_confusion(yt, meilleur.predict(Xt), f"Points de la main - {nom_meilleur}",
                          "confusion_points.png")
    final = meilleur
    if test_personne is not None or rapide:
        # la personne de test n'a pas servi à l'entraînement : pour le direct, on
        # réentraîne le modèle final sur TOUTES les données
        Xf, yf = augmenter(X, y)
        final = classifieurs(rapide)[nom_meilleur]
        final.fit(Xf, yf)
    config.DOSSIER_MODELES.mkdir(exist_ok=True)
    joblib.dump(final, config.DOSSIER_MODELES / "points.joblib")
    log("Modèle des points enregistré.")
    return {"scores": scores, "meilleur": nom_meilleur, "precision": meilleure_acc,
            "lettres": lettres, "n": len(X)}


def augmenter_gestes(X, y, n=5, graine=0):
    """Copies bruitées des gestes, un peu agrandies ou rétrécies (on a peu de gestes)."""
    rng = np.random.default_rng(graine)
    Xs, ys = [X], [y]
    for _ in range(n):
        echelle = rng.uniform(0.85, 1.15, size=(len(X), 1))
        Xb = X.copy()
        nt = X.shape[1] - 63            # partie trajectoire
        Xb[:, :nt] *= echelle
        Xb += rng.normal(0, 0.03, Xb.shape)
        Xs.append(Xb.astype(np.float32))
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def entrainer_gestes(test_personne=None, log=print):
    """Entraîne le modèle des gestes (J, Z, autre) et l'enregistre dans
    modeles/gestes.joblib. Renvoie un dictionnaire de résultats ou None."""
    X, y, pers = charger_sequences()
    if len(X) == 0:
        log("Gestes : aucun geste enregistré (J, Z) pour l'instant.")
        return None
    classes = [CLASSES_GESTE[k] for k in np.unique(y)]
    log(f"Gestes : {len(X)} gestes ({', '.join(classes)})")
    if len(classes) < 2:
        log("Gestes : il faut au moins 2 sortes de gestes (par ex. J et « autre »).")
        return None
    if test_personne is not None and test_personne not in set(pers.tolist()):
        test_personne = None
    a, t = decouper(pers, y, test_personne)
    if len(a) == 0:
        test_personne = None
        a, t = decouper(pers, y)
    modeles = {
        "kNN (k=3)": make_pipeline(StandardScaler(), KNeighborsClassifier(3)),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier((64,), max_iter=1000, random_state=0)),
    }
    Xa, ya = augmenter_gestes(X[a], y[a])
    nom_meilleur, acc_max = None, -1
    for nom, clf in modeles.items():
        clf.fit(Xa, ya)
        acc = float((clf.predict(X[t]) == y[t]).mean()) if len(t) else float("nan")
        log(f"Gestes {nom:9s} précision test = {100 * acc:.1f} %")
        if nom_meilleur is None or acc > acc_max:
            nom_meilleur, acc_max = nom, acc
    final = modeles[nom_meilleur]
    Xf, yf = augmenter_gestes(X, y)
    final.fit(Xf, yf)                    # modèle final : toutes les données
    config.DOSSIER_MODELES.mkdir(exist_ok=True)
    joblib.dump({"modele": final, "classes": CLASSES_GESTE},
                config.DOSSIER_MODELES / "gestes.joblib")
    log(f"Modèle des gestes enregistré ({nom_meilleur}).")
    return {"precision": acc_max, "meilleur": nom_meilleur}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-personne", default=None)
    args = p.parse_args()
    entrainer(args.test_personne)
    entrainer_gestes(args.test_personne)


if __name__ == "__main__":
    main()
