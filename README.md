# 🧠 Laboratorio Experimental de Reconocimiento Facial

Este proyecto implementa y compara tres métodos clásicos y modernos de **reconocimiento facial**:  
**DNN (Deep Neural Network con dlib), HAAR+LBPH y HOG+SVM**, integrados en una sola aplicación Python (`app.py`).

El laboratorio permite capturar imágenes desde la cámara, realizar la verificación facial y generar métricas experimentales de rendimiento, precisión y condiciones ambientales.

---

## ⚙️ Métodos Implementados

### 🧩 1. DNN.py
- Usa [`face_recognition`](https://github.com/ageitgey/face_recognition) (basado en *dlib*, una CNN ligera).  
- **Flujo:** carga rostros → codifica → compara mediante distancia euclidiana.  
- Alta precisión en entornos controlados.  
- Ideal para medir **confianza y distancia de embeddings**.

### 📸 2. HAAR.py
- Usa **Haar Cascade + LBPH (Local Binary Patterns Histograms)**.  
- **Flujo:** detección con cascada → entrenamiento LBPH → predicción con confianza.  
- Muy rápido, aunque menos robusto a variaciones de luz o ángulo.

### 📊 3. HOG.py
- Usa **Histogram of Oriented Gradients (HOG)** + **SVM**.  
- **Flujo:** extracción de características → entrenamiento/prueba → métricas estadísticas.  
- Excelente como método base para comparación experimental.

---

## 🎯 Objetivo

Crear una **aplicación unificada** (`app.py`) que:
- Capture una fotografía desde la webcam.  
- Ejecute la verificación del rostro con **DNN**, **HAAR+LBPH** y **HOG**.  
- Calcule métricas comparativas:

| Métrica | Descripción |
|----------|--------------|
| Exactitud (Accuracy) | Proporción de aciertos en detección |
| Precisión y Recall | Medidas de calidad del reconocimiento |
| Tiempo promedio | Velocidad del método |
| Falsos positivos/negativos | Tasas de error |
| Confianza promedio | Grado de similitud entre rostros |

---

## 🧠 Arquitectura del Sistema

### `app.py`
1. Captura fotos de **referencia** y **verificación** con `cv2.VideoCapture`.
2. Evalúa secuencialmente los tres métodos.
3. Mide tiempo, coincidencia y métricas adicionales.
4. Guarda los resultados en un **CSV** (`resultados_metricas.csv`).
5. Genera **gráficos comparativos** con `matplotlib`.

### Salida esperada

| Método | Tiempo (s) | Accuracy | Precision | Recall | F1 | Confianza Promedio |
|---------|-------------|----------|------------|--------|---------------------|
| DNN     | 0.42        | 0.98     | 0.96       | 0.97   | 0.96                |
| HAAR    | 0.15        | 0.82     | 0.80       | 0.77   | 0.79                |
| HOG     | 1.20        | 0.90     | 0.89       | 0.88   | 0.89                |

---

## 📈 Métricas Experimentales Adicionales

| Métrica | Descripción | Cálculo |
|----------|--------------|----------|
| **Confianza DNN** | Distancia entre embeddings | `face_distance` de `face_recognition` |
| **Diferencia HOG** | Distancia euclidiana entre vectores HOG | `np.linalg.norm()` |
| **Luminosidad media** | Nivel promedio de brillo (0–255) | `np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))` |
| **Coincidencia visual (SSIM)** | Similaridad estructural entre imágenes | `skimage.metrics.structural_similarity` |

Estas métricas permiten correlacionar condiciones ambientales (luz, distancia, cámara) con los resultados de reconocimiento facial.

---

## ⚡ Flujo de Ejecución Dinámico

1. Captura automática de imágenes de **referencia** y **verificación**.  
2. Evaluación mediante los tres métodos (DNN, HAAR, HOG).  
3. Cálculo de métricas y registro de condiciones experimentales.  
4. Visualización en `matplotlib` (barras comparativas).  
5. Registro automático en `resultados_metricas.csv`.

El CSV almacena:
FechaHora | Método | Tiempo | Match | Confianza | Luminancia_Ref | Luminancia_Ver | Contexto


Ejemplo de contexto:  
> “Luz natural, distancia 1 m, cámara Logitech HD 720p”

---

## 🧩 Instalación y Configuración

### 1️⃣ Descargar el clasificador Haar
Guarda el siguiente archivo en el mismo directorio del proyecto:  
👉 [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)

---

### 2️⃣ Instalar **CMake**
Descarga e instala desde [https://cmake.org/download/](https://cmake.org/download/)  
Selecciona **Windows x64 Installer (.msi)**  
Durante la instalación, activa la opción:
> *Add CMake to the system PATH for all users*

Verifica:

cmake --version

---

### 3️⃣ Instalar Visual Studio Build Tools

Descárgalo desde:
👉 https://visualstudio.microsoft.com/visual-cpp-build-tools/

Selecciona el paquete:

Desktop development with C++

Esto instala el compilador necesario para compilar dlib.

---

### 4️⃣ Crear y activar entorno virtual
CREAR
python -m venv .venv
ACTIVAR
.venv\Scripts\Activate.ps1

5️⃣ Instalar dependencias

Asegúrate de actualizar los paquetes base:

pip install --upgrade pip setuptools wheel


Luego instala los módulos del proyecto:

pip install dlib
pip install face_recognition opencv-python opencv-contrib-python scikit-image matplotlib numpy

---

### ▶️ Ejecución del programa

Dentro del entorno virtual:

python app.py

---

### 🧾 Resultados y Registro

Cada ejecución genera:

Dos fotos con fecha y hora (referencia_YYYYMMDD_HHMMSS.jpg, verificacion_YYYYMMDD_HHMMSS.jpg).

Registro automático en resultados_metricas.csv con:

Método utilizado

Tiempo de ejecución

Coincidencia y confianza

Nivel de luminancia

Contexto experimental

Esto permite analizar el rendimiento bajo diferentes condiciones de iluminación, distancia y cámara.

---

### 📊 Visualización Experimental

El programa genera automáticamente dos gráficos:

Tiempo de ejecución (en segundos)

Nivel de confianza promedio (0–1)

Ambos permiten comparar el comportamiento relativo de los métodos.

---

### 🧪 Conclusión

Este laboratorio es una herramienta experimental para medir, comparar y comprender el comportamiento de distintos métodos de reconocimiento facial bajo diversas condiciones.

Permite construir datasets personalizados, correlacionar métricas con entornos reales y profundizar en el análisis de eficiencia, precisión y robustez de los algoritmos faciales.

---

Lenguaje: Python 3.11+
Dependencias clave: opencv-python, face_recognition, scikit-image, matplotlib, numpy
