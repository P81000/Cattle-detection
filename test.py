from ultralytics import YOLO

model = YOLO("/home/ubuntu/Cattle-detection/runs/detect/train3/weights/best.pt")

source = "finalcounttest.mp4"

results = model(source, save=True, imgsz=320, conf=0.5)
