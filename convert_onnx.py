from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-seg.pt")  # load an official model
model = YOLO("../runs/segment/train/weights/best.pt")  # load a custom trained model
model.to("cuda")

# Export the model
model.export(device=0,
             format="onnx")