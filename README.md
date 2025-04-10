# Proyecto YOLOv6 en Vivo

Este proyecto utiliza YOLOv6 para realizar detección de objetos en tiempo real usando la webcam. Se emplea un modelo preentrenado (por ejemplo, `yolov6s.pt`) y se implementa el preprocesamiento mediante *letterbox* para mantener la relación de aspecto de la imagen. Además, se ajustan las coordenadas de las detecciones para que se dibujen correctamente sobre la imagen original.

## 🚀 Características

- **Detección en vivo:** Captura vídeo desde la webcam y muestra las detecciones en tiempo real.
- **Preprocesamiento con letterbox:** Redimensiona la imagen manteniendo la relación de aspecto, agregando *padding* según sea necesario.
- **Ajuste de coordenadas:** Las coordenadas de las detecciones se transforman para corresponder con la imagen original.
- **Safe globals:** Se utilizan *safe globals* para la carga segura de pesos de YOLOv6.

## 🛠 Instalación

### Requisitos

- Python 3.7 o superior  
- PyTorch  
- OpenCV (`opencv-python`)  
- NumPy  

### Instalación de dependencias

1. Crea un entorno virtual (opcional pero recomendado):

   ```bash
   python -m venv venv
   ```

2. Activa el entorno virtual:

   - En **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - En **Linux/Mac**:
     ```bash
     source venv/bin/activate
     ```

3. Instala las dependencias. Si tienes un archivo `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   O instala manualmente:

   ```bash
   pip install torch torchvision opencv-python numpy
   ```

## 📁 Estructura del proyecto

```
/TuProyecto
 ├── .gitignore
 ├── README.md
 ├── main.py
 ├── requirements.txt
 ├── YOLOv6/          # Repositorio oficial o submódulo de YOLOv6
 └── weights/
      └── yolov6s.pt  # Pesos preentrenados (descargados de la fuente oficial)
```

## ▶️ Uso

### Detección en vivo

Ejecuta el script principal:

```bash
python main.py
```

Selecciona el número de cámara cuando se te indique. Se abrirá una ventana que muestra las detecciones en tiempo real.

### Notas útiles

- **Umbral de confianza (`conf_thres`)**  
  Ajusta este valor en el diccionario `args` en `main.py`.  
  Valores bajos (ej. 0.05) detectan más objetos pero con más falsos positivos.  
  Valores altos (ej. 0.5+) filtran detecciones débiles.

- **Ajuste visual de los recuadros:**  
  Si los recuadros aparecen desalineados, revisa la función de transformación inversa de coordenadas respecto al *letterbox*.

- **Safe Globals:**  
  El código incluye todas las clases necesarias con `torch.serialization.add_safe_globals()` para evitar errores al cargar pesos en PyTorch 2.6+.

## 📌 Submódulos y Git

Si estás utilizando YOLOv6 como submódulo:

```bash
git submodule update --init --recursive
```

Si en cambio incluiste YOLOv6 directamente (copiado), asegúrate de **eliminar su .git interno** para evitar conflictos:

```bash
rm -rf YOLOv6/.git
```

## ✅ Conclusión

Este proyecto es ideal para aprender y experimentar con detección de objetos usando YOLOv6 en vivo. Está diseñado para ser claro, modular y fácilmente extensible. Si deseas personalizar el flujo o trabajar con tus propios datos, ¡tenés una base sólida para empezar!
