#!/usr/bin/env python3

import torch
import torch.onnx
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
model.eval()
dummy_input = torch.randn(1, 3, 416, 416)
torch.onnx.export(model.model, dummy_input, "yolov8n.onnx", opset_version=11)
