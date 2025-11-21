from ultralytics import YOLO

# Cargar el modelo
model = YOLO("runs/segment/train/weights/best.pt")  # Cargar el modelo personalizado

# Realizar la predicción sobre la imagen
results = model("predict.png")  # Predicción en una imagen

# Acceder a los resultados
for result in results:
    # Mostrar la imagen sin las cajas delimitadoras, solo las máscaras
    result.plot(save=True, boxes=False, masks=True)  # Solo muestra las máscaras sin las cajas delimitadoras
