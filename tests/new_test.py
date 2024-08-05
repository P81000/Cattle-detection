import cv2
import time
from ultralytics import YOLO

# Inicializar o modelo YOLO
model = YOLO("/home/masteguin/Codes/Cattle-detection/runs/detect/train3/weights/best.pt")

# Caminho do vídeo de entrada e saída
source = "/home/masteguin/Codes/Cattle-detection/finalcounttest.mp4"
output_path = "/home/masteguin/Codes/Cattle-detection/finalcounttest_with_count.mp4"

# Iniciar a contagem do tempo
start_time = time.time()

# Abrir o vídeo de entrada
cap = cv2.VideoCapture(source)

# Obter informações do vídeo de entrada
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Definir codec e criar VideoWriter para o vídeo de saída
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Contador total de vacas
total_cows = 0

# Processar cada quadro do vídeo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cow_count = 0

    # Detectar vacas no quadro atual
    results = model(frame, imgsz=128, conf=0.6)  # Ajuste imgsz para ser múltiplo de 32

    # Contar o número total de vacas detectadas em todos os resultados
    for detection in results:
        cow_count += detection.boxes.cls.tolist().count(0)

    total_cows += cow_count
    # Adicionar contador no canto superior direito
    cv2.putText(frame, f'Total Cows: {total_cows}', (width - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Escrever o quadro no vídeo de saída
    out.write(frame)

# Liberar recursos
cap.release()
out.release()

# Calcular o tempo total de execução
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo total de execução: {execution_time:.2f} segundos")
