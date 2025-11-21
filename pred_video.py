import cv2
from ultralytics import solutions
import torch

# Abrir el video
cap = cv2.VideoCapture("strawberries.mp4")
assert cap.isOpened(), "Error reading video file"

# Obtener las propiedades del video
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Redimensionar el video a un tamaño fijo (si es necesario)
resize_width = 640  # El ancho al que quieres redimensionar el video
resize_height = 480  # La altura al que quieres redimensionar el video

# Configurar el escritor de video
video_writer = cv2.VideoWriter("instance-segmentation.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps,
                               (resize_width, resize_height))

# Inicializar la segmentación de instancias
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Habilitar CUDA si está disponible
isegment = solutions.InstanceSegmentation(
    show=True,  # Mostrar la salida
    model="runs/segment/train/weights/best.pt",  # Asegúrate de usar el modelo adecuado
    device=device  # Usar CUDA si está disponible
)

# Procesar el video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("El frame del video está vacío o el procesamiento ha finalizado.")
        break

    # Redimensionar el frame antes de pasarlo al modelo
    im0_resized = cv2.resize(im0, (resize_width, resize_height))

    # Realizar la inferencia con el modelo de segmentación
    results = isegment(im0_resized)

    # Escribir el resultado procesado en el archivo de salida
    video_writer.write(results.plot_im)

# Liberar los recursos
cap.release()
video_writer.release()
cv2.destroyAllWindows()
