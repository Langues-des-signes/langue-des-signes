"""Reconnaissance en direct avec la webcam (ou sur une vidéo enregistrée).

Utilisation :
    python live.py                              # webcam
    python live.py --methode points             # choisir la méthode au départ
    python live.py --video essai.mp4 --sortie resultat.mp4
Touches :
    m : changer de méthode (cnn-peau, cnn-mp, points)
    c : calibrer la couleur de peau (poser la paume au centre du cadre jaune)
    f : apprendre le fond (retirer la main du cadre jaune avant d'appuyer)
    g : masquer / garder le fond autour de la main pour le CNN
    d : afficher / cacher les étapes du traitement
    r : effacer le texte      retour arrière : effacer une lettre
    q ou échap : quitter
"""
import argparse
import time

import cv2
import numpy as np

import config
import pretraitement as pre
from reconnaissance import Reconnaisseur, Lissage, METHODES


def dessiner_interface(image, r, methode, lettre, part, lissage, fps, rec, debug):
    h, w = image.shape[:2]
    if methode == "cnn-peau":
        _, (x1, y1, x2, y2) = pre.extraire_zone(image)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if r["main"] is not None:
        rec.mains.dessiner(image, r["main"])
    if r["boite"] is not None:
        x1, y1, x2, y2 = r["boite"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # lettre reconnue et top 3
    if lettre is not None:
        cv2.putText(image, lettre, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.5,
                    (0, 255, 0), 6)
    if r["probas"] is not None:
        top = np.argsort(r["probas"])[::-1][:3]
        for i, k in enumerate(top):
            cv2.putText(image, f"{config.LETTRES[k]} {r['probas'][k]:.2f}",
                        (20, 150 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(image, f"{methode}  {fps:4.1f} img/s", (w - 290, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(image, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(image, lissage.texte[-40:] + "_", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if debug and r["etapes"] is not None:
        panneau = pre.panneau_debug(r["etapes"], hauteur=110)
        ph, pw = panneau.shape[:2]
        pw = min(pw, w)
        image[h - 50 - ph:h - 50, 0:pw] = panneau[:, :pw]
    return image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default=None, help="fichier vidéo au lieu de la webcam")
    p.add_argument("--sortie", default=None, help="enregistrer le résultat en .mp4")
    p.add_argument("--methode", default=None, choices=METHODES)
    p.add_argument("--sans-affichage", action="store_true", help="pour les tests")
    p.add_argument("--sans-mediapipe", action="store_true")
    args = p.parse_args()

    rec = Reconnaisseur(avec_mediapipe=not args.sans_mediapipe)
    dispo = rec.disponibles()
    if not dispo:
        print("Aucun modèle trouvé : lancez entrainer_cnn.py et/ou entrainer_points.py")
        return
    methode = args.methode if args.methode in dispo else dispo[-1]
    print("Méthodes disponibles :", dispo, "- méthode choisie :", methode)

    source = cv2.VideoCapture(args.video if args.video else args.camera)
    miroir = args.video is None
    fps_video = source.get(cv2.CAP_PROP_FPS) or 30
    ecrivain = None
    lissage = Lissage()
    debug = True
    t0 = time.time()
    n_images, fps, t_fps = 0, 0.0, time.time()
    durees = {m: [] for m in METHODES}

    while True:
        ok, image = source.read()
        if not ok:
            break
        if miroir:
            image = cv2.flip(image, 1)
        # sur une vidéo, on utilise le temps de la vidéo et pas l'horloge
        temps = n_images / fps_video if args.video else time.time() - t0

        debut = time.perf_counter()
        r = rec.predire(image, methode, temps * 1000)
        durees[methode].append(time.perf_counter() - debut)
        lettre, part = lissage.mettre_a_jour(r["probas"], temps)

        n_images += 1
        if time.time() - t_fps >= 1:
            fps = n_images / (time.time() - t0)
            t_fps = time.time()
        affichage = dessiner_interface(image.copy(), r, methode, lettre, part, lissage,
                                       fps, rec, debug)

        if args.sortie:
            if ecrivain is None:
                h, w = affichage.shape[:2]
                ecrivain = cv2.VideoWriter(args.sortie, cv2.VideoWriter_fourcc(*"mp4v"),
                                           fps_video, (w, h))
            ecrivain.write(affichage)
        if args.sans_affichage:
            continue

        cv2.imshow("Langue des signes", affichage)
        touche = cv2.waitKey(1) & 0xFF
        if touche in (27, ord("q")):
            break
        elif touche == ord("m"):
            methode = dispo[(dispo.index(methode) + 1) % len(dispo)]
            lissage.votes.clear()
        elif touche == ord("c"):
            zone, _ = pre.extraire_zone(image)
            rec.seuils = pre.calibrer_peau(zone)
            print("Calibration :", rec.seuils)
        elif touche == ord("f"):
            rec.fond = image.copy()
            print("Fond appris")
        elif touche == ord("g"):
            rec.masquer_fond = not rec.masquer_fond
        elif touche == ord("d"):
            debug = not debug
        elif touche == ord("r"):
            lissage.texte = ""
        elif touche == 8:
            lissage.texte = lissage.texte[:-1]
        # la croix de la fenêtre doit arrêter le programme
        try:
            if cv2.getWindowProperty("Langue des signes", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    source.release()
    if ecrivain is not None:
        ecrivain.release()
    cv2.destroyAllWindows()
    rec.fermer()
    print("Texte reconnu :", repr(lissage.texte))
    for m, d in durees.items():
        if d:
            print(f"{m:9s} : {1000 * np.mean(d):6.1f} ms par image en moyenne "
                  f"({len(d)} images)")


if __name__ == "__main__":
    main()
