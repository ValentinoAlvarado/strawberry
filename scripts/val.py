from ultralytics import YOLO

# Definir la función que realiza la validación
def validate_model():
    # Cargar tu modelo personalizado
    model = YOLO("../runs/segment/train/weights/best.pt")  # Reemplaza con la ruta a tu modelo

    # Validar el modelo
    metrics = model.val()  # Sin argumentos, usa el conjunto de datos y configuraciones predeterminadas

    # Obtener las métricas de mAP para detección y segmentación
    print("mAP (box) 50-95:", metrics.box.map)  # mAP@50-95 para detección
    print("mAP (box) 50:", metrics.box.map50)  # mAP@50 para detección
    print("mAP (box) 75:", metrics.box.map75)  # mAP@75 para detección
    print("mAP (box) por categoría:", metrics.box.maps)  # mAP@50-95 por categoría

    print("mAP (seg) 50-95:", metrics.seg.map)  # mAP@50-95 para segmentación
    print("mAP (seg) 50:", metrics.seg.map50)  # mAP@50 para segmentación
    print("mAP (seg) 75:", metrics.seg.map75)  # mAP@75 para segmentación
    print("mAP (seg) por categoría:", metrics.seg.maps)  # mAP@50-95 por categoría

# Asegurarse de que el código solo se ejecute en el proceso principal
if __name__ == '__main__':
    validate_model()
