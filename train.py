#!/usr/bin/env python3

from ultralytics import YOLO

model = YOLO('./yolov8n.pt')

model.train(
        data='data.yaml',
        epochs=100,
        imgsz=256,
        batch=32,
        augment=True,
        weight_decay=0.0015,
        patience=7,
        single_cls=True,
        dropout=0.15
        )

metrics = model.val(data='data.yaml')

print(metrics)
