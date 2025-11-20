import torch
from ultralytics import YOLO

# Cargar el modelo preentrenado
model = YOLO("../yolo11n-seg.pt")  # Cargar un modelo preentrenado (recomendado para entrenar)
model.to("cuda")

if __name__ == "__main__":
    # Configuración de entrenamiento
    model.train(
        data="data.yaml",
        device=0,             # Usar la GPU 0
        batch=16,             # Tamaño de batch reducido a 16
        epochs=100,           # Número de épocas
        workers=4,            # Número de trabajadores para la carga de datos
        amp=True,              # Habilitar AMP (Automatic Mixed Precision)
    )

    # Liberar memoria de la GPU al final de cada época para evitar problemas de memoria
    torch.cuda.empty_cache()
