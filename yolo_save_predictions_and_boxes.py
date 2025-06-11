import os
import cv2
from ultralytics import YOLO

def save_yolo_format(image_path, predictions, output_dir):
    """Salva as boxes no formato YOLO para o LabelImg."""
    image_name = os.path.basename(image_path)
    image_name_no_ext = os.path.splitext(image_name)[0]
    height, width, _ = cv2.imread(image_path).shape

    # Arquivo para salvar as boxes no formato YOLO
    yolo_label_path = os.path.join(output_dir, f"{image_name_no_ext}.txt")

    with open(yolo_label_path, 'w') as f:
        for box in predictions:
            class_id, x_center, y_center, w, h = box
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def save_predictions_summary(output_summary_file, image_name, predictions):
    """Salva as predições em um arquivo de resumo separado."""
    with open(output_summary_file, 'a') as f:
        f.write(f"{image_name}: {predictions}\n")


def predict_and_save(model_path, image_dir, output_dir):
    """Realiza predições com o modelo YOLOv8 e salva os resultados."""
    # Certifique-se de que os diretórios de saída existem
    os.makedirs(output_dir, exist_ok=True)

    # Carregar o modelo YOLO
    model = YOLO(model_path, task="detect")

    # # Arquivo para salvar o resumo das predições
    # output_summary_file = os.path.join(output_dir, "predictions_summary.txt")

    # # Limpar o arquivo de resumo se já existir
    # if os.path.exists(output_summary_file):
    #     open(output_summary_file, 'w').close()

    # Loop sobre as imagens
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)

        if not os.path.isfile(image_path):
            continue

        # Realizar a predição        
        results = model.predict(image_path, conf=0.6)

        # Obter as boxes e classes
        predictions = []
        for result in results:  # Iterar sobre os resultados
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = box

                # Converter coordenadas para formato YOLO
                width = x2 - x1
                height = y2 - y1
                x_center = x1 + width / 2
                y_center = y1 + height / 2

                # Normalizar para o tamanho da imagem
                img_height, img_width, _ = cv2.imread(image_path).shape
                x_center /= img_width
                y_center /= img_height
                width /= img_width
                height /= img_height

                predictions.append((int(class_id), x_center, y_center, width, height))

        # Salvar boxes no formato YOLO
        save_yolo_format(image_path, predictions, output_dir)

        # # Salvar resumo das predições
        # save_predictions_summary(output_summary_file, image_name, predictions)


if __name__ == "__main__":  

    # Caminho do modelo YOLO treinado    
    model_path = ...

    # Diretório com as imagens para predição
    image_dir = ...

    # Diretório para salvar os resultados
    output_dir = ...

    # Realizar predições e salvar resultados
    predict_and_save(model_path, image_dir, output_dir)