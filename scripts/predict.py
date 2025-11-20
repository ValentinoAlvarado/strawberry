from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Cargar el modelo
model = YOLO("../runs/segment/train/weights/best.pt")  # cargar el modelo personalizado

# Realizar la predicción sobre una imagen
results = model("predict.png")  # realizar la predicción sobre la imagen

# Acceder a los resultados
for result in results:
    # Para las máscaras (si se usan en el modelo)
    if result.masks:
        xy = result.masks.xy  # coordenadas de la máscara en formato de polígonos
        xyn = result.masks.xyn  # coordenadas normalizadas
        masks = result.masks.data  # máscara en formato de matriz (num_objetos x H x W)

    # Extraer los cuadros delimitadores (bounding boxes) y las puntuaciones de confianza
    boxes = result.boxes  # cajas delimitadoras
    confidences = boxes.conf  # puntuaciones de confianza
    labels = result.names  # nombres de las clases

    # Dibujar los cuadros delimitadores sobre la imagen
    img = result.plot()  # dibujar cuadros delimitadores y máscaras si están disponibles

# Mostrar la imagen con las predicciones
plt.imshow(img)
plt.axis('off')  # Para quitar los ejes
plt.show()

# Si prefieres guardar la imagen con las predicciones
result.save()  # Guarda la imagen con las predicciones en una carpeta (runs/predict)
