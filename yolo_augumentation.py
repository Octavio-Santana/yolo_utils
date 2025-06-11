import os
import cv2
import albumentations as A
from albumentations.augmentations.bbox_utils import convert_bbox_from_yolo, convert_bbox_to_yolo
import random

# Gerar N augmentations por imagem
N = 5

# Diretórios
IMAGES_DIR = "caminho/para/imagens_originais"
LABELS_DIR = "caminho/para/labels_yolo"
OUT_IMAGES_DIR = "caminho/para/imagens_aumentadas"
OUT_LABELS_DIR = "caminho/para/labels_aumentadas"

os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUT_LABELS_DIR, exist_ok=True)

# Definindo transformações com bounding boxes
transform = A.Compose([
    A.Rotate(limit=15, p=0.7),
    A.RandomBrightnessContrast(p=0.6),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=0, p=0.7),
    A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
    A.Perspective(scale=(0.02, 0.05), p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Função para ler bboxes YOLO
def load_yolo_labels(txt_path, img_width, img_height):
    boxes = []
    class_labels = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.strip().split())
            boxes.append([x, y, w, h])
            class_labels.append(int(cls))
    return boxes, class_labels

# Loop pelas imagens
for img_file in os.listdir(IMAGES_DIR):
    if not img_file.endswith(('.jpg', '.png')):
        continue

    img_path = os.path.join(IMAGES_DIR, img_file)
    label_path = os.path.join(LABELS_DIR, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))

    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    bboxes, class_labels = load_yolo_labels(label_path, w, h)

    # Gerar N augmentations por imagem
    for i in range(N):  # 5 versões aumentadas por imagem
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)

        aug_img = transformed['image']
        aug_bboxes = transformed['bboxes']
        aug_labels = transformed['class_labels']

        # Salvar nova imagem
        new_img_name = f"{os.path.splitext(img_file)[0]}_aug{i}.jpg"
        new_label_name = f"{os.path.splitext(img_file)[0]}_aug{i}.txt"
        cv2.imwrite(os.path.join(OUT_IMAGES_DIR, new_img_name), aug_img)

        # Salvar novo label
        with open(os.path.join(OUT_LABELS_DIR, new_label_name), 'w') as f:
            for cls, box in zip(aug_labels, aug_bboxes):
                x, y, w, h = box
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
