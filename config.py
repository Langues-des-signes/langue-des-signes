"""Réglages communs à tous les scripts du projet."""
from pathlib import Path

RACINE = Path(__file__).resolve().parent
DOSSIER_DONNEES = RACINE / "donnees"          # nos propres exemples (enregistrés par app.py)
DOSSIER_MODELES = RACINE / "modeles"          # modèles entraînés
DOSSIER_FIGURES = RACINE / "figures"          # figures pour le rapport
DOSSIER_SIGN_MNIST = RACINE / "sign_mnist"    # sign_mnist_train.csv et sign_mnist_test.csv

# Alphabet ASL. 24 lettres se font sans bouger (« statiques ») : ce sont les classes du CNN
# et du classifieur des points. J et Z demandent un mouvement : un second modèle les
# reconnaît à partir d'une suite d'images (voir mains.caracteristiques_sequence).
LETTRES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
LETTRES_MOUVEMENT = ['J', 'Z']
AUTRE = '_'                        # classe « autre mouvement » (ni J ni Z)
ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DOSSIER_REFERENCES = RACINE / "references"     # images d'exemple des 26 lettres

TAILLE_CNN = 28  # les images de Sign MNIST font 28 x 28

# Main utilisée pour signer : "gauche", "droite" ou "les deux".
# Pour l'instant on ne signe qu'avec la main GAUCHE : l'autre main est ignorée, ce qui
# évite de mélanger deux façons de signer et donne de meilleurs résultats.
MAIN_UTILISEE = "gauche"
# Si l'appli confond gauche et droite (elle dit « main droite » alors que c'est la
# gauche), mettre True.
INVERSER_GAUCHE_DROITE = True    # vérifié sur notre webcam (10/11)

# Modèle MediaPipe de détection des points de la main (téléchargé automatiquement)
URL_MODELE_MAIN = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                   "hand_landmarker/float16/latest/hand_landmarker.task")
FICHIER_MODELE_MAIN = DOSSIER_MODELES / "hand_landmarker.task"

# Seuils de peau dans YCbCr (valeurs classiques de Chai et Ngan, 1999).
# Ils sont remplacés par une calibration sur la main de l'utilisateur si on la fait.
CB_MIN, CB_MAX = 77, 127
CR_MIN, CR_MAX = 133, 173

# Zone de l'écran où on fait les signes (fractions de la largeur et hauteur).
# Le visage reste en dehors, ce qui évite qu'il soit pris pour une main.
ZONE = (0.55, 0.10, 0.95, 0.75)   # x1, y1, x2, y2

# Lissage dans le temps pour le direct
NB_IMAGES_VOTE = 10        # vote majoritaire sur les 10 dernières images
CONFIANCE_MIN = 0.6        # en dessous, on n'affiche pas de lettre
DUREE_VALIDATION = 0.8     # secondes de lettre stable pour l'ajouter au mot
DUREE_ESPACE = 1.5         # secondes sans main pour ajouter un espace

# Application (app.py)
NB_PAR_LETTRE = 100        # exemples enregistrés par lettre dans « Enregistrer des signes »
NB_CORRECTION = 40         # exemples enregistrés quand on corrige une lettre en direct
MAX_PAR_LETTRE = 300       # par séance : au-delà, on n'ajoute plus de corrections
NB_IMAGES_TEST = 45        # images analysées par lettre dans « Comparer les 3 moteurs »
DECOMPTE = 3               # secondes avant l'enregistrement d'une lettre

# Stockage : les points sont gardés pour chaque image, mais les images (qui ne servent
# qu'à affiner le CNN) seulement une sur IMAGE_TOUTES_LES, en JPEG de TAILLE_IMAGE pixels.
# Deux images qui se suivent sont presque identiques : en garder une sur 3 suffit.
IMAGE_TOUTES_LES = 3
TAILLE_IMAGE = 64

# Gestes J et Z
DUREE_GESTE = 2.0          # secondes enregistrées pour un geste (les moments immobiles
                           # au début et à la fin sont retirés automatiquement)
FENETRE_GESTE = 2.5        # en direct : on garde les 2,5 dernières secondes de la main
SEUIL_IMMOBILE = 0.08      # la main est arrêtée si elle bouge moins que ça en 0,2 s
NB_REPETITIONS = 15        # répétitions du geste dans « Enregistrer des signes »
POINTS_GESTE = 10          # le geste est rééchantillonné sur 10 instants
SEUIL_MOUVEMENT = 0.35     # un geste doit déplacer la main d'au moins 0,35 taille de main
CONFIANCE_GESTE = 0.7      # probabilité minimale pour accepter J ou Z en direct
