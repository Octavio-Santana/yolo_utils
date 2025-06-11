import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

model_path = ...
model = YOLO(model=model_path, task="detect")

# Gerar cores RGB distintas para cada classe
def generate_colors(n):
    np.random.seed(42)  # Reprodutibilidade
    colors = np.random.randint(0, 255, size=(n, 3), dtype='uint8')
    return {i: tuple(int(c) for c in colors[i]) for i in range(n)}

class_names = model.names
# class_colors = generate_colors(len(class_names))

class_colors = {
    0: (255, 0, 0),       # Vermelho
    1: (0, 255, 0),       # Verde
    2: (0, 0, 255),       # Azul
    3: (255, 255, 0),     # Amarelo
    4: (255, 0, 255),     # Magenta
    5: (0, 255, 255),     # Ciano
    6: (128, 0, 255),     # Roxo
    7: (255, 128, 0),     # Laranja
    8: (0, 128, 255),     # Azul-claro
    9: (0, 0, 0),         # Preto
    10: (128, 128, 128),  # Cinza médio
}

def show_predictions(img_path, class_names=class_names, class_colors=class_colors):
    
    # Faz a predição
    results = model(img_path)[0]

    # Carrega a imagem original
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Desenha as caixas manualmente
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da bounding box
        cls_id = int(box.cls[0])                # ID da classe
        label = class_names[cls_id]             # Nome da classe
        color = class_colors[cls_id]
        
        # Desenhar retângulo
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Colocar texto (somente a classe, sem confiança)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)

    # Exibe a imagem
    plt.imshow(img)
    plt.axis('off')
    plt.show()