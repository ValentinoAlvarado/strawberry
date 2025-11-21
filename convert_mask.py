import cv2
import numpy as np
import os


# Función para convertir las coordenadas del polígono a formato YOLO (normalizado)
def convert_to_yolo_format(polygon, img_width, img_height):
    normalized_polygon = []
    for point in polygon:
        x, y = point
        x_normalized = x / img_width
        y_normalized = y / img_height
        normalized_polygon.append((x_normalized, y_normalized))
    return normalized_polygon


# Función principal para procesar las máscaras PNG y guardar el archivo de etiquetas YOLO
def process_mask(mask_image_path, output_txt_path, class_id=0):
    # Cargar la máscara PNG (en escala de grises, donde cada objeto tiene un valor único)
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    # Obtener el tamaño de la imagen
    img_height, img_width = mask_image.shape

    # Obtener las instancias únicas (todos los valores distintos de 0)
    unique_labels = np.unique(mask_image)

    # Crear el archivo de salida .txt
    with open(output_txt_path, "w") as f:
        for label in unique_labels:
            if label == 0:  # Ignorar el fondo (0)
                continue

            # Crear una máscara binaria para cada instancia
            instance_mask = np.where(mask_image == label, 255, 0).astype(np.uint8)

            # Encontrar los contornos de la instancia
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Convertir el contorno a una lista de coordenadas (polígono)
                polygon = contour.reshape(-1, 2)

                # Convertir las coordenadas del polígono a formato YOLO (normalizado)
                normalized_polygon = convert_to_yolo_format(polygon, img_width, img_height)

                # Guardar el ID de la clase y las coordenadas del polígono en el archivo .txt
                polygon_str = " ".join([f"{x} {y}" for x, y in normalized_polygon])
                f.write(f"{class_id} {polygon_str}\n")


# Directorio base
base_dir = 'StrawDI_Db1'

# Listado de las carpetas que contienen las máscaras
subdirs = ['train', 'val', 'test']

# Procesar cada subdirectorio
for subdir in subdirs:
    masks_dir = os.path.join(base_dir, subdir, 'masks')
    labels_dir = os.path.join(base_dir, subdir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)  # Crear la carpeta 'labels' si no existe

    # Recorrer todas las máscaras PNG en el directorio
    for mask_filename in os.listdir(masks_dir):
        if mask_filename.endswith('.png'):
            # Ruta completa de la máscara PNG
            mask_image_path = os.path.join(masks_dir, mask_filename)

            # Crear el archivo de texto correspondiente
            label_filename = os.path.splitext(mask_filename)[0] + '.txt'
            output_txt_path = os.path.join(labels_dir, label_filename)

            # Procesar la máscara y guardar el archivo .txt
            process_mask(mask_image_path, output_txt_path)

            print(f"Etiquetas guardadas en: {output_txt_path}")

print("Proceso completado.")
