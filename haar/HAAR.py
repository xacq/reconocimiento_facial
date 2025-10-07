import cv2
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Ruta al dataset
path = 'dataset'

# Crear el reconocedor LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Cargar el clasificador Haar Cascade
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Función para obtener imágenes y etiquetas
def get_images_and_labels(path):
    face_samples = []
    ids = []
    
    # Recorrer cada carpeta de persona en el dataset
    for person_folder in os.listdir(path):
        # Ignorar carpetas que no sean de personas (como .ipynb_checkpoints)
        if person_folder == '.ipynb_checkpoints':
            continue
            
        person_path = os.path.join(path, person_folder)
        if not os.path.isdir(person_path):
            continue
            
        # Obtener el ID de la persona (usamos el nombre de la carpeta como ID)
        try:
            person_id = int(person_folder)
        except ValueError:
            continue  # Si el nombre no es un número, lo saltamos
            
        # Recorrer cada imagen en la carpeta de la persona
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            
            # Verificar que sea un archivo de imagen (por extensión)
            if not (image_name.lower().endswith(('.png', '.jpg', '.jpeg'))):
                continue
                
            # Convertir a escala de grises
            pil_image = Image.open(image_path).convert('L')
            img_numpy = np.array(pil_image, 'uint8')
            
            # Detectar rostros
            faces = detector.detectMultiScale(img_numpy)
            
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(person_id)
    
    return face_samples, ids

# Obtener datos de entrenamiento
faces, ids = get_images_and_labels(path)

# Entrenar el modelo
recognizer.train(faces, np.array(ids))

# Guardar el modelo entrenado
recognizer.write('trainer.yml')
print("Modelo entrenado y guardado como 'trainer.yml'")


# Cargar el clasificador Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Cargar el modelo LBPH entrenado
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# Diccionario de nombres (debe coincidir con los IDs del entrenamiento)
names = {1: "Persona1", 2: "Persona2"}  # Ajusta según tus IDs

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros con Haar Cascade
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        # Dibujar rectángulo
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Predecir quién es con LBPH
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        # Verificar confianza (menor es mejor)
        if confidence < 100:
            name = names.get(id, "Desconocido")
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "Desconocido"
            confidence_text = f"{round(100 - confidence)}%"
        
        # Mostrar nombre y confianza
        cv2.putText(frame, f"{name} {confidence_text}", (x+5, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Mostrar imagen
    cv2.imshow('Reconocimiento Facial', frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()