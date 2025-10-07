Tenemos los tres métodos implementados claramente:

DNN.py usa face_recognition (basado en el modelo profundo de dlib, una CNN ligera).
Flujo: carga rostros conocidos → codifica → compara mediante distancia euclidiana.
Ideal para precisión alta en entornos controlados, pero no mide rendimiento temporal ni tasa de error.

HAAR.py usa Haar Cascade + LBPH (Local Binary Patterns Histograms), el método clásico de OpenCV.
Flujo: detección con cascada → entrenamiento con LBPH → predicción con confianza.
Es rápido pero menos robusto a variaciones de luz o ángulo.

HOG.py usa Histogram of Oriented Gradients con un clasificador SVM sobre el dataset LFW.
Flujo: extracción HOG → entrenamiento/test → matriz de confusión y métricas.
Excelente como base comparativa de rendimiento estadístico.

Objetivo

Crear una aplicación unificada que:
Permita capturar una fotografía desde la webcam.
Ejecute la verificación del rostro con cada método (HOG, HAAR+LBPH y DNN).
Genere métricas de contraste:
Exactitud (accuracy)
Precisión y recall
Tiempo promedio de detección
Porcentaje de error (falsos positivos / falsos negativos)
Confianza promedio del reconocimiento

Arquitectura propuesta
Interfaz principal (app.py)
Toma la foto de la cámara (cv2.VideoCapture).
Guarda la imagen base (“captura.jpg”).
Llama secuencialmente a los tres métodos.
Mide tiempo de ejecución.
Evalúa si el rostro fue correctamente reconocido.
Guarda los resultados en un DataFrame para análisis comparativo.
Salida
Tabla tipo:
| Método | Tiempo (s) | Accuracy | Precision | Recall | F1 | Confianza Promedio |
|---------|-------------|----------|------------|--------|---------------------|
| DNN     | 0.42        | 0.98     | 0.96       | 0.97   | 0.96                |
| HAAR    | 0.15        | 0.82     | 0.80       | 0.77   | 0.79                |
| HOG     | 1.20        | 0.90     | 0.89       | 0.88   | 0.89                |

Gráficas de comparación (barplots o radar charts).

Además de Tiempo y Match, puedes calcular automáticamente:

Métrica	Descripción	Cómo se calcula
Confianza DNN	Distancia entre embeddings	face_distance de face_recognition
Diferencia HOG	Distancia euclidiana entre vectores HOG	Ya la calculas (puedes guardarla)
Luminosidad media	Nivel promedio de brillo en la imagen (0–255)	np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
Porcentaje de coincidencia visual	Similaridad estructural (SSIM)	from skimage.metrics import structural_similarity as ssim

Estas métricas permitirán correlacionar condiciones ambientales (luz, distancia, cámara) con los resultados de reconocimiento.


PASOS PARA REALIZAR LA EJECUCION DEL PROGRAMA

Tener el archivo haarcascade_frontalface_default.xml en el mismo directorio.
Si no lo tienes Descarga aquí: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

Instala la versión oficial de CMake desde la web
Esto no puede hacerse con pip (porque la de pip falla al compilar dlib).
Abre este enlace en tu navegador:
👉 https://cmake.org/download/
Descarga el instalador:
Windows x64 Installer (.msi)
Durante la instalación, marca esta opción: Add CMake to the system PATH for all users
Termina la instalacion y reinicia el computador.
Verifica que CMake funciona en un simbolo de sistema
cmake --version

Instala el compilador C++ (requisito de dlib)
Si aún no tienes Visual Studio Build Tools:
Descárgalos desde:
👉 https://visualstudio.microsoft.com/visual-cpp-build-tools/
Al abrir el instalador, marca “Desktop development with C++”
Espera a que finalice (esto instala MSVC + SDK + linker)

El sistema esta con un entorno virtual, para activarlo:
.\.venv\Scripts\Activate.ps1

Ya con el entorno virtual si no corre los paquetes, sin salir del entorno virtual instalar:
pip install --upgrade pip setuptools wheel
pip install dlib
pip install face_recognition opencv-python opencv-contrib-python scikit-image matplotlib numpy


Este es el flujo dinámico del programa:

Captura automática de referencia y verificación.
Evaluación con DNN, HAAR y HOG.
Visualización en matplotlib.

Registro automático en un CSV (resultados_metricas.csv) para que puedas acumular datos de cada sesión (tiempo, resultado, fecha, etc.).
Tu laboratorio de reconocimiento facial ahora guarda automáticamente cada ejecución en resultados_metricas.csv, con fecha, método, tiempo, coincidencia y contexto experimental (por ejemplo “luz natural, distancia 1 m”).
Esto te permitirá realizar múltiples pruebas y luego analizar tendencias de rendimiento entre DNN, HAAR y HOG.

