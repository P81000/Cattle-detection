#!/usr/bin/env python3

import onnxruntime as ort
import numpy as np

# Carregar o modelo ONNX
session = ort.InferenceSession('yolov8n_quantized.onnx')

# Preparar entrada (substitua por entrada real)
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)

# Executar inferência
outputs = session.run(None, {input_name: input_data})

print("Inferência executada com sucesso.")
