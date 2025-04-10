import sys
import os
import cv2
import time
import torch
import numpy as np

# Añadir carpeta YOLOv6 al path
sys.path.append(os.path.join(os.getcwd(), 'YOLOv6'))

# ============================
# IMPORTAR CLASES SEGURAS
# ============================
import torch.serialization
from torch.nn import ReLU, Identity, Sequential, Conv2d, BatchNorm2d, ModuleList, SiLU
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.conv import ConvTranspose2d

# Importar clases de YOLOv6
from yolov6.layers.common import (
    SimSPPF, ConvModule, CSPSPPFModule, ConvBNReLU, ConvBNSiLU, BiFusion, Transpose
)
from yolov6.models.yolo import Model
from yolov6.models.efficientrep import EfficientRep
from yolov6.layers.common import RepVGGBlock, SimCSPSPPF, RepBlock
from yolov6.models.reppan import RepBiFPANNeck
from yolov6.models.heads.effidehead_distill_ns import Detect

# Registrar safe globals para la carga de pesos
torch.serialization.add_safe_globals([
    Model,
    EfficientRep,
    RepVGGBlock,
    SimCSPSPPF,
    RepBlock,
    ConvModule,
    CSPSPPFModule,
    ConvBNReLU,
    ConvBNSiLU,
    RepBiFPANNeck,
    BiFusion,
    MaxPool2d,
    Transpose,
    ConvTranspose2d,
    Detect,
    ModuleList,
    SiLU,
    ReLU,
    Identity,
    Sequential,
    Conv2d,
    BatchNorm2d,
    SimSPPF
])

# ============================
# IMPORTAR YOLOv6
# ============================
from yolov6.core.inferer import Inferer
from yolov6.utils.nms import non_max_suppression

# ============================
# FUNCIONES AUXILIARES DE PREPROCESAMIENTO
# ============================

def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Redimensiona la imagen manteniendo la relación de aspecto mediante letterbox.
    Agrega padding para obtener una imagen de new_shape.
    """
    shape = image.shape[:2]  # (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    image_resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image_padded

def preprocess_image(image, target_size=640):
    """
    Aplica letterbox para redimensionar la imagen a target_size x target_size,
    convierte la imagen de BGR a RGB, la normaliza y la convierte a tensor.
    """
    # Aplicar letterbox
    img_letterboxed = letterbox(image, new_shape=(target_size, target_size))
    # (Opcional) Mostrar la imagen letterboxed para verificar visualmente
    cv2.imshow("Imagen letterboxed", img_letterboxed)
    cv2.waitKey(1)
    # Convertir de BGR a RGB
    image_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
    # Normalizar a [0,1]
    image_normalized = image_rgb.astype(np.float32) / 255.0
    # Convertir a tensor y reorganizar a formato (B, C, H, W)
    tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    print("Tensor preprocesado shape:", tensor.shape)  # Debería ser [1, 3, target_size, target_size]
    return tensor

# ============================
# SELECCIÓN DE CÁMARA
# ============================

def listar_camaras(max_camaras=5):
    """Lista los índices de cámaras disponibles."""
    disponibles = []
    for i in range(max_camaras):
        cap = cv2.VideoCapture(i)
        ret, _ = cap.read()
        if ret:
            disponibles.append(i)
            cap.release()
    return disponibles

print("🔍 Buscando cámaras disponibles...")
camaras_disponibles = listar_camaras()
if not camaras_disponibles:
    print("❌ No se encontraron cámaras.")
    sys.exit(0)
print("🎥 Cámaras disponibles:")
for cam in camaras_disponibles:
    print(f"[{cam}] Cámara {cam}")

while True:
    try:
        cam_index = int(input("Selecciona el número de cámara que deseas usar: "))
        if cam_index in camaras_disponibles:
            break
        else:
            print("Número inválido. Intenta con uno de los mostrados.")
    except ValueError:
        print("Por favor, ingresa un número válido.")

# ============================
# CONFIGURACIÓN DEL MODELO
# ============================

args = {
    'weights': 'weights/yolov6s.pt',
    'device': 'cpu',  # o 'cuda:0'
    'yaml': 'YOLOv6/data/coco.yaml',
    'img_size': 640,
    'half': False,
    'conf_thres': 0.25,  # Reducido para captar detecciones débiles
    'iou_thres': 0.45,
    'classes': None,
    'agnostic_nms': False,
    'max_det': 1000,
}

# ============================
# INICIALIZAR MODELO
# ============================
# Para la webcam, se establece webcam=True.
inferer = Inferer(
    source=cam_index,
    weights=args['weights'],
    device=args['device'],
    yaml=args['yaml'],
    img_size=args['img_size'],
    half=args['half'],
    webcam=True,
    webcam_addr=''
)

# ============================
# CAPTURA DE VIDEO DESDE LA CÁMARA
# ============================
cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    sys.exit(0)
print("✅ Presiona 'q' para salir.")

# ============================
# LOOP PRINCIPAL PARA VIDEO EN VIVO
# ============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesar el frame: se usa nuestra función que aplica letterbox y transforma a tensor
    img_tensor = preprocess_image(frame, target_size=args['img_size'])
    # También se puede obtener la imagen letterboxed para dibujo (aunque en este ejemplo usamos el frame original)
    frame_letterboxed = letterbox(frame, new_shape=(args['img_size'], args['img_size']))

    start_time = time.time()
    # Obtener salida cruda del modelo
    pred_results = inferer.model(img_tensor)
    print("Shape de pred_results:", pred_results.shape)
    # Imprimir la confianza máxima (índice 4) para ver si las predicciones tienen algún valor alto
    print("Confianza máxima:", torch.max(pred_results[..., 4]))

    # Aplicar NMS
    detections = non_max_suppression(
        pred_results,
        args['conf_thres'],
        args['iou_thres'],
        args['classes'],
        args['agnostic_nms'],
        max_det=args['max_det']
    )[0]
    end_time = time.time()
    fps = f"{1 / (end_time - start_time):.2f}" if (end_time - start_time) > 0 else "∞"

    if detections is not None and len(detections):
        print(f"✅ Detecciones en el frame: {len(detections)}")
        # Convertir detecciones a numpy (si es tensor)
        detections = detections.cpu().detach().numpy()

        for det in detections:
            # Cada detección es [x1, y1, x2, y2, conf, cls]
            xyxy = det[:4]
            conf = det[4]
            cls = int(det[5])
            label = f"{inferer.class_names[cls]} {conf:.2f}"
            inferer.plot_box_and_label(
                frame,  # Imagen sobre la que dibujar
                max(round(sum(frame.shape) / 2 * 0.003), 2),
                xyxy,
                label,
                color=inferer.generate_colors(cls, True)
            )

    else:
        print("⚠️ No se detectó ningún objeto en el frame.")

    cv2.imshow("YOLOv6 - Cámara en vivo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
