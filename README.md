# Detección de Objetos con MobileNetV3 - Unidad 4, Parte B

## 📌 Descripción general

Este parte del trabajo implementa un sistema de detección de objetos en video utilizando la red neuronal **MobileNetV3**. Está desarrollado en C++ y usa OpenCV para el procesamiento de imágenes y video.

La solución permite comparar el rendimiento del modelo ejecutándose con CPU y, opcionalmente, con GPU, evaluando el impacto en la velocidad de detección (FPS).

---

## 🧠 ¿Qué es MobileNetV3?

**MobileNetV3** es una red neuronal convolucional optimizada para dispositivos con recursos limitados (como móviles o sistemas embebidos). En esta implementación, se utiliza una versión preentrenada para detectar múltiples objetos comunes en escenas del mundo real (personas, autos, etc.).

---

## 🎯 ¿Qué hace este código?

- Carga un video desde archivo
- Preprocesa cada fotograma para adaptarlo al tamaño de entrada del modelo
- Ejecuta la detección de objetos en tiempo real usando la red MobileNetV3 
- Dibuja las cajas delimitadoras y las etiquetas de los objetos detectados
- Muestra el video procesado en una ventana con el conteo de FPS (cuadros por segundo)

---

## 🧪 Pruebas de rendimiento

El sistema fue probado ejecutándose con CPU. También puede adaptarse para correr con GPU, midiendo la diferencia de rendimiento en términos de FPS.

---

## 📁 Estructura del proyecto

├── principal.cpp # Código fuente principal
├── models/
│ ├── mobileNetV3.onnx # Modelo preentrenado
│ ├── coco.names # Archivo de etiquetas
├── video4.mp4 # Video de entrada para pruebas
├── vision.bin # Binario compilado
├── Makefile / CMakeLists.txt
└── README.md # Este archivo

---

## ⚙️ Requisitos

- Linux (Ubuntu probado)
- OpenCV (con soporte CUDA si deseas usar GPU)
- CMake (para compilar)
- Compilador compatible con C++17

---

## 🚀 Compilación

```bash
mkdir build
cd build
cmake ..
make
./principal
```

## 🎥 Resultados

La aplicación muestra el video procesado en tiempo real, indicando:

- Objetos detectados.

- Rendimiento (FPS).

- Comparación CPU vs GPU.
