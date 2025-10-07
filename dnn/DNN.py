import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_known_faces(folder_path):
    known_encodings = []
    known_names = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            
            face_encoding = face_recognition.face_encodings(image)
            
            if face_encoding:
                known_encodings.append(face_encoding[0])
                known_names.append(os.path.splitext(filename)[0])
    
    return known_encodings, known_names

def recognize_faces(image_path, known_encodings, known_names, tolerance=0.6):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        min_distance = np.min(distances)
        
        if min_distance <= tolerance:
            name = known_names[np.argmin(distances)]
        else:
            name = "Desconocido"
        
        face_names.append(name)
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(image, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
    
    return image

known_faces_path = "known_faces"
known_encodings, known_names = load_known_faces(known_faces_path)

print(f"Rostros cargados: {len(known_encodings)}")
print("Nombres:", known_names)

test_image_path = "test_image.jpg"
result_image = recognize_faces(test_image_path, known_encodings, known_names)

# Mostrar resultado
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
