from ultralytics import YOLO

model = YOLO("/home/masteguin/Codes/Cattle-detection/runs/detect/train3/weights/best.pt")

source = "/home/masteguin/Codes/Cattle-detection/finalcounttest.mp4"

results = model(source, save=True, imgsz=96, conf=0.7)
