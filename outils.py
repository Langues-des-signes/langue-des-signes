"""Fonctions partagées : lecture de nos données, découpage entraînement / test, figures."""
import csv

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

FICHIER_POINTS = config.DOSSIER_DONNEES / "points.csv"          # lettres statiques
FICHIER_SEQUENCES = config.DOSSIER_DONNEES / "sequences.csv"    # gestes J, Z et « autre »
DOSSIER_IMAGES = config.DOSSIER_DONNEES / "images"              # pour affiner le CNN
CLASSES_GESTE = config.LETTRES_MOUVEMENT + [config.AUTRE]


def image_vers_28(image_bgr):
    """Image couleur de la main -> 28 x 28 en niveaux de gris, histogramme étiré."""
    gris = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    petit = cv2.resize(gris, (28, 28), interpolation=cv2.INTER_AREA)
    return (petit - petit.min()) / (petit.max() - petit.min() + 1e-6)


def decouper(personnes, y, test_personne=None, part_test=0.2, graine=0):
    """Indices d'entraînement et de test.

    Si test_personne est donné, toutes ses données vont dans le test : c'est le test le
    plus honnête (le modèle n'a jamais vu cette main). Sinon, 20 % au hasard dans
    chaque lettre.
    """
    personnes = np.asarray(personnes)
    if test_personne is not None:
        test = personnes == test_personne
        return np.nonzero(~test)[0], np.nonzero(test)[0]
    rng = np.random.default_rng(graine)
    test = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.nonzero(y == c)[0]
        rng.shuffle(idx)
        test[idx[:max(1, int(part_test * len(idx)))]] = True
    return np.nonzero(~test)[0], np.nonzero(test)[0]


def charger_points():
    """Lit donnees/points.csv. Renvoie X (n x 63), y (n), personnes (n)."""
    X, y, pers = [], [], []
    with open(FICHIER_POINTS, encoding="utf-8") as f:
        for ligne in csv.reader(f):
            if len(ligne) != 65 or ligne[1] not in config.LETTRES:
                continue   # ligne abîmée : on l'ignore
            pers.append(ligne[0])
            y.append(config.LETTRES.index(ligne[1]))
            X.append([float(v) for v in ligne[2:]])
    return np.array(X, np.float32).reshape(-1, 63), np.array(y, int), np.array(pers)


def charger_sequences():
    """Lit donnees/sequences.csv. Renvoie X, y (indice dans CLASSES_GESTE), personnes."""
    X, y, pers = [], [], []
    if FICHIER_SEQUENCES.exists():
        with open(FICHIER_SEQUENCES, encoding="utf-8") as f:
            for ligne in csv.reader(f):
                if len(ligne) < 3 or ligne[1] not in CLASSES_GESTE:
                    continue
                pers.append(ligne[0])
                y.append(CLASSES_GESTE.index(ligne[1]))
                X.append([float(v) for v in ligne[2:]])
    if not X:
        return np.zeros((0, 0), np.float32), np.zeros(0, int), np.array([], dtype=str)
    return np.array(X, np.float32), np.array(y, int), np.array(pers)


def charger_images_perso(test_personne=None):
    """Lit donnees/images/<lettre>/<séance>_*.jpg (ou .png) et prépare des images 28 x 28."""
    x, y, pers = [], [], []
    for k, lettre in enumerate(config.LETTRES):
        dossier = DOSSIER_IMAGES / lettre
        if not dossier.exists():
            continue
        for f in sorted(list(dossier.glob("*.jpg")) + list(dossier.glob("*.png"))):
            img = cv2.imread(str(f))
            if img is None:
                continue
            x.append(image_vers_28(img))
            y.append(k)
            pers.append(f.name.split("_")[0])
    if not x:
        empty = np.zeros((0, 28, 28, 1), np.float32)
        return (empty, np.zeros(0, int)), (empty, np.zeros(0, int)), []
    x = np.array(x, np.float32)[..., None]
    y = np.array(y)
    a, t = decouper(pers, y, test_personne)
    return (x[a], y[a]), (x[t], y[t]), sorted(set(pers))


def matrice_confusion(y_vrai, y_pred, titre, nom_fichier):
    """Enregistre la matrice de confusion (en %) dans figures/."""
    n = len(config.LETTRES)
    m = np.zeros((n, n))
    for a, b in zip(y_vrai, y_pred):
        m[a, b] += 1
    lignes = m.sum(1, keepdims=True)
    m_pct = np.divide(m, lignes, out=np.zeros_like(m), where=lignes > 0) * 100
    plt.figure(figsize=(9, 8))
    plt.imshow(m_pct, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(label="%")
    plt.xticks(range(n), config.LETTRES)
    plt.yticks(range(n), config.LETTRES)
    plt.xlabel("lettre prédite")
    plt.ylabel("vraie lettre")
    acc = (np.asarray(y_vrai) == np.asarray(y_pred)).mean() * 100
    plt.title(f"{titre} - précision {acc:.1f} %")
    for i in range(n):
        for j in range(n):
            if m_pct[i, j] >= 5 and i != j:
                plt.text(j, i, f"{m_pct[i, j]:.0f}", ha="center", va="center",
                         fontsize=7, color="red")
    plt.tight_layout()
    config.DOSSIER_FIGURES.mkdir(exist_ok=True)
    plt.savefig(config.DOSSIER_FIGURES / nom_fichier, dpi=120)
    plt.close()
    return acc


def enregistrer_exemple(image_bgr, main, lettre, personne, numero, seance):
    """Enregistre un exemple de lettre statique.

    Les 63 caractéristiques des points vont dans points.csv à chaque fois ; l'image de la
    main (petit JPEG) seulement une fois sur config.IMAGE_TOUTES_LES, car deux images qui
    se suivent sont presque identiques. Renvoie False si la main est trop petite.
    """
    import mains
    x1, y1, x2, y2 = main["boite"]
    if x2 - x1 < 20 or y2 - y1 < 20:
        return False
    config.DOSSIER_DONNEES.mkdir(exist_ok=True)
    with open(FICHIER_POINTS, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([personne, lettre] +
                               [f"{v:.5f}" for v in mains.caracteristiques(main)])
    if numero % config.IMAGE_TOUTES_LES == 0:
        dossier = DOSSIER_IMAGES / lettre
        dossier.mkdir(parents=True, exist_ok=True)
        morceau = cv2.resize(image_bgr[y1:y2, x1:x2], (config.TAILLE_IMAGE,) * 2,
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dossier / f"{personne}_{seance}_{numero:05d}.jpg"), morceau,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True


def enregistrer_sequence(caracteristiques, lettre, personne):
    """Enregistre un geste (J, Z ou « autre ») dans sequences.csv."""
    config.DOSSIER_DONNEES.mkdir(exist_ok=True)
    with open(FICHIER_SEQUENCES, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([personne, lettre] + [f"{v:.5f}" for v in caracteristiques])


def compter_exemples():
    """Nombre d'exemples par personne et par lettre : {personne: {lettre: n}}.
    Pour J, Z et « autre », c'est le nombre de gestes enregistrés."""
    comptes = {}
    for fichier in (FICHIER_POINTS, FICHIER_SEQUENCES):
        if not fichier.exists():
            continue
        with open(fichier, encoding="utf-8") as f:
            for ligne in csv.reader(f):
                if len(ligne) >= 2:
                    d = comptes.setdefault(ligne[0], {})
                    d[ligne[1]] = d.get(ligne[1], 0) + 1
    return comptes


def supprimer(personne, lettre=None):
    """Supprime les exemples d'une personne (d'une seule lettre si lettre est donnée) :
    lignes des fichiers CSV et images. Renvoie le nombre d'exemples supprimés."""
    n = 0
    for fichier in (FICHIER_POINTS, FICHIER_SEQUENCES):
        if not fichier.exists():
            continue
        with open(fichier, encoding="utf-8") as f:
            lignes = list(csv.reader(f))
        garder = [l for l in lignes if not (len(l) >= 2 and l[0] == personne
                                            and (lettre is None or l[1] == lettre))]
        n += len(lignes) - len(garder)
        if len(garder) != len(lignes):
            temporaire = fichier.with_suffix(".tmp")
            with open(temporaire, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(garder)
            temporaire.replace(fichier)
    if DOSSIER_IMAGES.exists():
        dossiers = [DOSSIER_IMAGES / lettre] if lettre else list(DOSSIER_IMAGES.iterdir())
        for d in dossiers:
            if d.is_dir():
                for f in list(d.glob(f"{personne}_*")):
                    f.unlink()
    return n


def tout_supprimer():
    """Efface TOUS les signes enregistrés (toutes séances) et les modèles appris à partir
    d'eux. Le CNN de base (cnn.keras), le modèle MediaPipe, les images d'exemple et les
    figures sont gardés. Renvoie le nombre d'exemples effacés."""
    import shutil
    n = sum(sum(d.values()) for d in compter_exemples().values())
    for fichier in (FICHIER_POINTS, FICHIER_SEQUENCES):
        if fichier.exists():
            fichier.unlink()
    if DOSSIER_IMAGES.exists():
        shutil.rmtree(DOSSIER_IMAGES)
    for nom in ("points.joblib", "gestes.joblib", "cnn_affine.keras"):
        f = config.DOSSIER_MODELES / nom
        if f.exists():
            f.unlink()
    return n
