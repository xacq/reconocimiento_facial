import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
import csv
from datetime import datetime
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
from skimage.metrics import structural_similarity as ssim

# ===============================
# CONFIGURACIÓN
# ===============================
RESULTADOS_CSV = 'resultados_metricas.csv'

# ===============================
# 1. CAPTURAR FOTO DESDE WEBCAM
# ===============================
def capturar_foto(nombre_base):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{nombre_base}_{timestamp}.jpg"
    cam = cv2.VideoCapture(0)
    print(f"Presiona 'c' para capturar {nombre_archivo}, 'q' para salir.")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error al acceder a la cámara.")
            break
        cv2.imshow('Captura', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cv2.imwrite(nombre_archivo, frame)
            print(f"Fotografía guardada como {nombre_archivo}")
            break
        elif key == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    return nombre_archivo

# ===============================
# 2. CÁLCULO DE LUMINANCIA
# ===============================
def calcular_luminancia(imagen_path):
    img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    return round(np.mean(img), 2)

# ===============================
# 3. MÉTODO DNN (face_recognition)
# ===============================
def evaluar_dnn(ref_path, test_path):
    inicio = time.time()
    ref_img = face_recognition.load_image_file(ref_path)
    test_img = face_recognition.load_image_file(test_path)

    ref_enc = face_recognition.face_encodings(ref_img)
    test_enc = face_recognition.face_encodings(test_img)

    if len(ref_enc) == 0 or len(test_enc) == 0:
        print("[DNN] No se detectaron rostros en alguna imagen.")
        return {'Método': 'DNN', 'Tiempo': time.time() - inicio, 'Match': 0, 'Confianza': 0}

    result = face_recognition.compare_faces([ref_enc[0]], test_enc[0])[0]
    distance = face_recognition.face_distance([ref_enc[0]], test_enc[0])[0]
    confianza = round(1 - distance, 3)

    tiempo = time.time() - inicio
    return {'Método': 'DNN', 'Tiempo': tiempo, 'Match': int(result), 'Confianza': confianza}

# ===============================
# 4. MÉTODO HAAR
# ===============================
def evaluar_haar(ref_path, test_path):
    inicio = time.time()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or test_img is None or face_cascade.empty():
        print("[HAAR] Error al cargar imágenes o clasificador.")
        return {'Método': 'HAAR', 'Tiempo': 0, 'Match': 0, 'Confianza': 0}

    ref_faces = face_cascade.detectMultiScale(ref_img, 1.1, 4)
    test_faces = face_cascade.detectMultiScale(test_img, 1.1, 4)

    tiempo = time.time() - inicio
    match = int(len(ref_faces) > 0 and len(test_faces) > 0)
    confianza = round(np.clip(len(test_faces) / (len(ref_faces) + 1), 0, 1), 3)
    return {'Método': 'HAAR', 'Tiempo': tiempo, 'Match': match, 'Confianza': confianza}

# ===============================
# 5. MÉTODO HOG
# ===============================
def evaluar_hog(ref_path, test_path):
    inicio = time.time()
    ref_img = cv2.imread(ref_path)
    test_img = cv2.imread(test_path)
    if ref_img is None or test_img is None:
        return {'Método': 'HOG', 'Tiempo': 0, 'Match': 0, 'Confianza': 0, 'Distancia_HOG': 0}

    ref_gray = rgb2gray(resize(ref_img, (50, 50)))
    test_gray = rgb2gray(resize(test_img, (50, 50)))

    ref_fd = hog(ref_gray, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), block_norm='L2-Hys')
    test_fd = hog(test_gray, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm='L2-Hys')

    distancia = np.linalg.norm(ref_fd - test_fd)
    umbral = 0.4
    match = int(distancia < umbral)
    confianza = round(max(0, 1 - distancia), 3)

    tiempo = time.time() - inicio
    return {'Método': 'HOG', 'Tiempo': tiempo, 'Match': match,
            'Confianza': confianza, 'Distancia_HOG': round(distancia, 4)}

# ===============================
# 6. GUARDAR RESULTADOS EN CSV
# ===============================
def guardar_resultados_csv(resultados, contexto, ref_lum, ver_lum):
    encabezados = ['FechaHora', 'Método', 'Tiempo', 'Match', 'Confianza', 
                   'Distancia_HOG', 'Luminancia_Ref', 'Luminancia_Ver', 'Contexto']
    archivo_existe = False
    try:
        open(RESULTADOS_CSV, 'r')
        archivo_existe = True
    except FileNotFoundError:
        archivo_existe = False

    with open(RESULTADOS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=encabezados)
        if not archivo_existe:
            writer.writeheader()
        for r in resultados:
            writer.writerow({
                'FechaHora': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Método': r['Método'],
                'Tiempo': round(r['Tiempo'], 3),
                'Match': r['Match'],
                'Confianza': r.get('Confianza', 0),
                'Distancia_HOG': r.get('Distancia_HOG', 0),
                'Luminancia_Ref': ref_lum,
                'Luminancia_Ver': ver_lum,
                'Contexto': contexto
            })

# ===============================
# 7. MAIN
# ===============================
def main():
    print("==============================")
    print(" LABORATORIO EXPERIMENTAL DE RECONOCIMIENTO FACIAL AVANZADO")
    print("==============================")

    ref_img = capturar_foto("referencia")
    ver_img = capturar_foto("verificacion")

    contexto = input("\nDescribe condiciones de prueba (luz, distancia, cámara, etc.): ")

    ref_lum = calcular_luminancia(ref_img)
    ver_lum = calcular_luminancia(ver_img)

    print(f"\nLuminancia promedio -> Referencia: {ref_lum} | Verificación: {ver_lum}")

    resultados = []
    print("\nEjecutando DNN...")
    resultados.append(evaluar_dnn(ref_img, ver_img))

    print("\nEjecutando HAAR...")
    resultados.append(evaluar_haar(ref_img, ver_img))

    print("\nEjecutando HOG...")
    resultados.append(evaluar_hog(ref_img, ver_img))

    guardar_resultados_csv(resultados, contexto, ref_lum, ver_lum)

    # Visualización
    metodos = [r['Método'] for r in resultados]
    tiempos = [r['Tiempo'] for r in resultados]
    confianzas = [r['Confianza'] for r in resultados]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(metodos, tiempos, color='skyblue')
    plt.title('Tiempo de ejecución (segundos)')
    plt.ylabel('Segundos')

    plt.subplot(1, 2, 2)
    plt.bar(metodos, confianzas, color='lightgreen')
    plt.title('Confianza promedio (0–1)')
    plt.ylabel('Confianza')

    plt.suptitle('Comparación Experimental de Métodos de Reconocimiento Facial')
    plt.tight_layout()
    plt.show()

    print("\nResultados obtenidos:")
    for r in resultados:
        print(r)

if __name__ == '__main__':
    main()
