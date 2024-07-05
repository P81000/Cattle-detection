import cv2

# Leitura do vídeo AVI
input_video_path = "object_counting_output.avi"
output_video_path = "object_counting_output.mp4"

cap = cv2.VideoCapture(input_video_path)
assert cap.isOpened(), "Erro ao abrir o arquivo de vídeo."

# Obter propriedades do vídeo
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Inicializar o escritor de vídeo para MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

# Liberação dos objetos de captura e escrita
cap.release()
out.release()
cv2.destroyAllWindows()

