from ultralytics import YOLO

model = YOLO('runs/detect/train5/weights/best.pt')
metrics = model.val(data='data.yaml')
print(metrics)
