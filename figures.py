"""Figure des étapes du traitement d'image, pour le rapport.

Utilisation :
    python figures.py photo.jpg                 # sur une photo
    python figures.py                           # prend une photo avec la webcam (espace)
    python figures.py photo.jpg --calibrer      # seuils de peau appris sur la photo
"""
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import pretraitement as pre


def prendre_photo(camera=0):
    cam = cv2.VideoCapture(camera)
    while True:
        ok, image = cam.read()
        image = cv2.flip(image, 1)
        affichage = image.copy()
        _, (x1, y1, x2, y2) = pre.extraire_zone(image)
        cv2.rectangle(affichage, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.imshow("Espace pour prendre la photo", affichage)
        if cv2.waitKey(1) & 0xFF == ord(" "):
            break
    cam.release()
    cv2.destroyAllWindows()
    return image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?")
    p.add_argument("--calibrer", action="store_true")
    p.add_argument("--zone-entiere", action="store_true",
                   help="utiliser toute l'image (photo déjà centrée sur la main)")
    p.add_argument("--sortie", default="etapes_pretraitement.png")
    args = p.parse_args()

    image = cv2.imread(args.image) if args.image else prendre_photo()
    zone = (0, 0, 1, 1) if args.zone_entiere else config.ZONE
    seuils = pre.SeuilsPeau()
    if args.calibrer:
        seuils = pre.calibrer_peau(pre.extraire_zone(image, zone)[0])
    r = pre.chaine_complete(image, seuils, zone=zone)
    ycrcb = cv2.cvtColor(r["zone"], cv2.COLOR_BGR2YCrCb)

    vues = [
        (cv2.cvtColor(r["zone"], cv2.COLOR_BGR2RGB), "1. Zone d'intérêt", None),
        (ycrcb[:, :, 2], "2a. Composante Cb", "gray"),
        (ycrcb[:, :, 1], "2b. Composante Cr", "gray"),
        (r["peau"], f"3. Seuillage peau\n{seuils}", "gray"),
        (r["median"], "4. Filtre médian", "gray"),
        (r["morphologie"], "5. Ouverture + fermeture", "gray"),
        (r["main"], "6. Plus grande composante", "gray"),
        (r["entree_cnn"] if r["entree_cnn"] is not None else np.zeros((28, 28)),
         "7. Entrée du CNN (28 x 28)", "gray"),
    ]
    plt.figure(figsize=(16, 8))
    for i, (img, titre, cmap) in enumerate(vues):
        plt.subplot(2, 4, i + 1)
        plt.imshow(img, cmap=cmap)
        plt.title(titre, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    config.DOSSIER_FIGURES.mkdir(exist_ok=True)
    plt.savefig(config.DOSSIER_FIGURES / args.sortie, dpi=130)

    # histogrammes de Cb et Cr avec les seuils de peau
    plt.figure(figsize=(10, 3.5))
    for i, (c, nom, bas, haut) in enumerate([(2, "Cb", seuils.cb_min, seuils.cb_max),
                                             (1, "Cr", seuils.cr_min, seuils.cr_max)]):
        plt.subplot(1, 2, i + 1)
        plt.hist(ycrcb[:, :, c].ravel(), bins=256, range=(0, 255))
        plt.axvspan(bas, haut, color="red", alpha=0.2, label="intervalle peau")
        plt.title(f"Histogramme de {nom}")
        plt.legend()
    plt.tight_layout()
    plt.savefig(config.DOSSIER_FIGURES / ("histo_" + args.sortie), dpi=130)
    print("Figures enregistrées dans", config.DOSSIER_FIGURES)


if __name__ == "__main__":
    main()
