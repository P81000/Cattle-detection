#!/usr/bin/env python3

from ultralytics import YOLO

model = YOLO('yolov8n.yaml').load('yolov8n.pt')

model.train(
        data='data.yaml',
        epochs=50,
        imgsz=320,
        batch=16,
        weight_decay=0.0015,
        patience=5,
        single_cls=True,
        dropout=0.25,
        device=0
        )

metrics = model.val(data='data.yaml')

print(metrics)
