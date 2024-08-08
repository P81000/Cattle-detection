#!/usr/bin/env python3

import torch
import torch.quantization as quantization

# 1. Carregar o modelo salvo
model = torch.load('besttest.pt')
model.eval()

# 2. Configurar quantização padrão
model.qconfig = quantization.get_default_qconfig('fbgemm')

# 3. Preparar o modelo para quantização
prepared_model = quantization.prepare(model)

# 4. Calibração do modelo
# Exemplo de um conjunto de dados de calibração
# Substitua pelo seu próprio conjunto de dados
input_data = torch.randn(1, 3, 320, 320)  # Substitua pelas dimensões reais da entrada do seu modelo

# Passar dados de calibração pelo modelo preparado
prepared_model(input_data)

# 5. Converter o modelo preparado para um modelo quantizado
quantized_model = quantization.convert(prepared_model)

# 6. Salvar o modelo quantizado
torch.save(quantized_model.state_dict(), 'quantized_best_static.pt')

print("Modelo quantizado salvo com sucesso.")
