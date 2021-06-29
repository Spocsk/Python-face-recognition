# Python-face-recognition


# Reconnaissance faciale - Python

Ce projet Python a pour but de pouvoir reconnaître facilement des visages humain par le biais d'une caméra. Ce projet a été programmer avec l'aide de OpenCV2 et de la librairie Python face_recognition.


## Requis

Pour pouvoir lancer correctement le script il est nécessaire d'avoir les librairies suivantes Numpy - face_recognition et OpenCv2. Pour les 2 premières librairies il faut lancer la commande suivante dans un terminal :
```bash
pip install -r requirements.txt
```
par la suite il vous faudra installer OpenCv2,  dans un terminal lancer la commande suivante:
```bash
sudo apt update
sudo apt install python3-opencv
```
Ensuite il faudra installer cmake
```bash
sudo apt install cmake
```
>**Remarque**: si cela ne marche toujours pas c'est que vous n'avez votre cmake dans vos variables d'environnement.

## Explication

Tout d'abord pour pouvoir être reconnu par le script, il faut que vous placiez une photo de la personne que vous voulez autorisé.

![enter image description here](https://zupimages.net/up/21/26/xczd.png)


![enter image description here](https://zupimages.net/up/21/26/j1ey.png)

>Maintenant passons au code.

```python
import  face_recognition
import cv2
import  numpy  as  np
```
dans cette section nous importons les différentes librairies nécessaires.

```python
# Prend comme référence la caméra numéro 0 ( celle par défaut )

video_capture = cv2.VideoCapture(0)

# Charge une image et en détermine comment la reconnaitre

biden_image = face_recognition.load_image_file("images/biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
dylan_image = face_recognition.load_image_file("images/dylan.png")
dylan_face_encoding = face_recognition.face_encodings(dylan_image)[0]
elon_image = face_recognition.load_image_file("images/elon.jpg")
elon_face_encoding = face_recognition.face_encodings(elon_image)[0]

# Création de tableaux des visages encodés et leurs noms.

known_face_encodings = [
	biden_face_encoding,
	dylan_face_encoding,
	elon_face_encoding
]

known_face_names = [
	"Joe Biden",
	"Dylan",
	"Elon"
]
```

Dans cette première section on initialise les différentes variables qui nous serviront pour la suite du code.

>**Premièrement**

On définis dois définir qu'on utilise notre caméra et pas seulement une photo. Pour ce faire nous utilisons le périphérique par défaut qui est donc normalement la caméra principale de notre pc.

>**Par la suite**

On charge les images qui vont nous servir à reconnaître les personnes autorisées.  Une fois les images chargées, on encode les images pour pouvoir en retirer un tableau de valeur qui détermine les spécificitées de chaque visages. Une fois tout cela fait on place le tout dans des tableaux pour pouvoir mieux les utiliser par la suite.

Dans la deuxième partie va servir pour le traitement de l'image et par la suite afficher l'image complète sur l'écran.

```python
while  True:

  

# Capture une seul frame de la caméra

ret, frame = video_capture.read()

  
  

# Réduis la taille du carré de détection d'image ( pour une reconnaisance plus rapide )

small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

  

# Convertissez l'image de la couleur BGR (utilisée par OpenCV) en couleur RVB (utilisée par face_recognition)

rgb_small_frame = small_frame[:, :, ::-1]

  

# Ne traitez qu'une image sur deux de la vidéo pour gagner du temps

if  process_this_frame:

# Trouver tous les visages dans l'image actuelle de la vidéo

face_locations = face_recognition.face_locations(rgb_small_frame)

face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

  

face_names = []

for  face_encoding  in  face_encodings:

# Voir si le visage correspond au(x) visage(s) connu(s)

matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

name = "Unknown"

  

# Trouve la similitude entre les visage connus et les visages sur la vidéo.

face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

best_match_index = np.argmin(face_distances)

if  matches[best_match_index]:

name = known_face_names[best_match_index]
face_names.append(name)
process_this_frame = not  process_this_frame
```

>**Enfin** on affiche le résultat

```python
  

# Afficher le résultat

for (top, right, bottom, left), name  in  zip(face_locations, face_names):

# Redimensionnez les emplacements des visages puisque le cadre dans lequel nous avons détecté a été redimensionné à 1/4 de taille

top *= 4

right *= 4

bottom *= 4

left *= 4

  

# Dessine un rectangle autour des visages

cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

  

# Desinne un label à côté du rectangle

cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

font = cv2.FONT_HERSHEY_DUPLEX

cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

  

# Afficher l'image complète

cv2.imshow('Video', frame)

  

# Pressez la touche 'q' pour quitter

if  cv2.waitKey(1) & 0xFF == ord('q'):

break

  

# Libère la caméra

video_capture.release()

cv2.destroyAllWindows()
```

## Poursuite

L'objectif final est de pouvoir mettre le script python dans une petite carte d'electronique embarquée t-elle qu'une raspberry pi pour pouvoir ensuite y connecter une caméra portatif et de pouvoir rendre le projet nomade. Une fois tout ceci assemblé le circuit global sera installé en hauteur au-dessus d'une porte pour pouvoir identifier les personnes qui ne sont pas enregistrées dans la base de donnée ou bien celle qui le sont.

![enter image description here](https://cdn-blog.adafruit.com/uploads/2018/12/Facial-Recognition-My-Circuit-Setup-Fig6-570x480.png)
