"""Entraîne le CNN sur Sign Language MNIST (base publique de 27 455 images de mains), puis, si on a
enregistré nos propres exemples avec app.py, l'affine sur nos images.

Points importants :
  - augmentation de données (rotation, zoom, décalage, contraste, miroir), car les
    images de la webcam ne ressemblent pas exactement à celles de Sign MNIST
  - arrêt automatique quand la précision de validation ne progresse plus
  - deuxième phase d'entraînement sur nos propres images (« fine-tuning »)

Utilisation :
    python entrainer_cnn.py                      # Sign MNIST seul
    python entrainer_cnn.py --affiner            # + nos images (donnees/images)
    python entrainer_cnn.py --affiner --test-personne s20261110143000
"""
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from keras import layers

import config
from outils import matrice_confusion, charger_images_perso


def charger_sign_mnist():
    def lire(nom):
        d = np.loadtxt(config.DOSSIER_SIGN_MNIST / nom, delimiter=",", skiprows=1)
        y = d[:, 0].astype(int)
        x = (d[:, 1:] / 255.0).reshape(-1, 28, 28, 1).astype(np.float32)
        y[y > 9] -= 1   # la classe 9 (J) n'existe pas : on décale les suivantes
        return x, y
    return lire("sign_mnist_train.csv"), lire("sign_mnist_test.csv")


def creer_modele(nb_classes=len(config.LETTRES)):
    """Deux couches de convolution puis deux couches denses, avec une couche
    d'augmentation de données au début (active seulement pendant l'entraînement)."""
    augmentation = keras.Sequential([
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.12),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.3),
        layers.RandomFlip("horizontal"),   # pour les gauchers
    ], name="augmentation")
    modele = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        augmentation,
        layers.Conv2D(32, 5, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 5, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(nb_classes, activation="softmax"),
    ])
    modele.compile(optimizer=keras.optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modele


def entrainer(modele, x, y, xv, yv, epoques, nom, lot=128, log=None, patience=15):
    rappels = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience,
                                             restore_best_weights=True)]
    if log is not None:
        rappels.append(keras.callbacks.LambdaCallback(
            on_epoch_end=lambda e, l: log(f"époque {e + 1}/{epoques} : entraînement "
                                          f"{100 * l['accuracy']:.0f} %, test "
                                          f"{100 * l['val_accuracy']:.0f} %")))
    hist = modele.fit(x, y, epochs=epoques, batch_size=lot, validation_data=(xv, yv),
                      callbacks=rappels, verbose=2 if log is None else 0)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history["loss"], label="entraînement")
    plt.plot(hist.history["val_loss"], label="test")
    plt.title("Coût"); plt.xlabel("époque"); plt.legend(); plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(hist.history["accuracy"], label="entraînement")
    plt.plot(hist.history["val_accuracy"], label="test")
    plt.title("Précision"); plt.xlabel("époque"); plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig(config.DOSSIER_FIGURES / f"courbes_{nom}.png", dpi=120)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epoques", type=int, default=100)
    p.add_argument("--affiner", action="store_true",
                   help="continuer l'entraînement sur nos images (donnees/images)")
    p.add_argument("--test-personne", default=None,
                   help="personne gardée pour le test lors de l'affinage")
    p.add_argument("--repartir", action="store_true",
                   help="ne pas réentraîner sur Sign MNIST, partir de modeles/cnn.keras")
    args = p.parse_args()
    config.DOSSIER_FIGURES.mkdir(exist_ok=True)
    config.DOSSIER_MODELES.mkdir(exist_ok=True)
    fichier = config.DOSSIER_MODELES / "cnn.keras"

    modele = None
    if args.repartir and fichier.exists():
        modele = keras.models.load_model(fichier)
    else:
        (x, y), (xt, yt) = charger_sign_mnist()
        print("Sign MNIST :", x.shape, xt.shape)
        modele = creer_modele()
        modele.summary()
        entrainer(modele, x, y, xt, yt, args.epoques, "sign_mnist")
        _, acc = modele.evaluate(xt, yt, verbose=0)
        print(f"Précision sur le test de Sign MNIST : {acc:.4f}")
        matrice_confusion(yt, modele.predict(xt, verbose=0).argmax(1),
                          "Sign MNIST (test)", "confusion_cnn_sign_mnist.png")
        modele.save(fichier)

    if args.affiner:
        affiner(args.test_personne, args.epoques, modele=modele)


def affiner(test_personne=None, epoques=40, log=print, modele=None):
    """Continue l'entraînement du CNN sur nos images (donnees/images).
    Renvoie un dictionnaire avec la précision avant et après, ou None."""
    config.DOSSIER_FIGURES.mkdir(exist_ok=True)
    if modele is None:
        fichier = config.DOSSIER_MODELES / "cnn.keras"
        if not fichier.exists():
            log("modeles/cnn.keras introuvable.")
            return None
        modele = keras.models.load_model(fichier)
    (x, y), (xt, yt), personnes = charger_images_perso(test_personne)
    if len(x) == 0 or len(xt) == 0:
        log("Pas assez d'images dans donnees/images : enregistrez d'abord des lettres.")
        return None
    log(f"Nos images : {len(x)} pour l'entraînement, {len(xt)} pour le test "
        f"(personnes : {', '.join(personnes)})")
    _, acc0 = modele.evaluate(xt, yt, verbose=0)
    log(f"Précision AVANT affinage : {100 * acc0:.1f} %")
    # petits lots (on a peu d'images) et taux d'apprentissage un peu plus petit
    modele.compile(optimizer=keras.optimizers.Adam(5e-4),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    entrainer(modele, x, y, xt, yt, epoques, "affinage", lot=32,
              log=None if log is print else log, patience=10)
    _, acc1 = modele.evaluate(xt, yt, verbose=0)
    log(f"Précision APRÈS affinage : {100 * acc1:.1f} %")
    matrice_confusion(yt, modele.predict(xt, verbose=0).argmax(1),
                      "CNN affiné (nos images, test)", "confusion_cnn_affine.png")
    modele.save(config.DOSSIER_MODELES / "cnn_affine.keras")
    log("CNN affiné enregistré.")
    with open(config.DOSSIER_FIGURES / "resultats_cnn.csv", "a", newline="") as f:
        csv.writer(f).writerow(["affinage", test_personne, len(x), len(xt), acc0, acc1])
    return {"avant": acc0, "apres": acc1}


if __name__ == "__main__":
    main()
