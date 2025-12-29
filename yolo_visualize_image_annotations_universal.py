from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics.data.utils import segments2boxes


def visualize_image_annotations_universal(
    image_path,
    txt_path,
    label_map=None,
    draw_boxes=True,
    draw_segments=True,
    box_color=(255, 0, 0),
    segment_color=(0, 255, 0),
    line_width=2,
):
    """
    Visualiza anotações YOLO Detection e YOLO Segmentation automaticamente.

    Args:
        image_path (str): Caminho da imagem
        txt_path (str): Caminho do label (.txt)
        label_map (dict): {class_id: class_name}
        draw_boxes (bool): Desenha bbox (para detecção ou segmentação)
        draw_segments (bool): Desenha polígonos (somente segmentação)
    """

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size

    with open(txt_path, encoding="utf-8") as file:
        for line in file:
            parts = list(map(float, line.split()))
            class_id = int(parts[0])
            label = label_map.get(class_id, str(class_id)) if label_map else str(class_id)

            # ============================
            # YOLO DETECTION FORMAT
            # ============================
            if len(parts) == 5:
                _, x, y, w, h = parts

                x1 = (x - w / 2) * img_width
                y1 = (y - h / 2) * img_height
                x2 = (x + w / 2) * img_width
                y2 = (y + h / 2) * img_height

                if draw_boxes:
                    draw.rectangle(
                        [x1, y1, x2, y2],
                        outline=box_color,
                        width=line_width,
                    )

                draw.text((x1, y1), label, fill=box_color)

            # ============================
            # YOLO SEGMENTATION FORMAT
            # ============================
            elif len(parts) > 5:
                coords = np.array(parts[1:]).reshape(-1, 2)

                # Normalizado → pixel
                polygon = [
                    (x * img_width, y * img_height)
                    for x, y in coords
                ]

                if draw_segments:
                    draw.polygon(
                        polygon,
                        outline=segment_color,
                        width=line_width,
                    )

                if draw_boxes:
                    bbox = segments2boxes([coords])[0]
                    bx, by, bw, bh = bbox

                    x1 = (bx - bw / 2) * img_width
                    y1 = (by - bh / 2) * img_height
                    x2 = (bx + bw / 2) * img_width
                    y2 = (by + bh / 2) * img_height

                    draw.rectangle(
                        [x1, y1, x2, y2],
                        outline=box_color,
                        width=line_width,
                    )

                draw.text((polygon[0][0], polygon[0][1]), label, fill=segment_color)

            else:
                raise ValueError(f"Formato de label inválido: {line}")

    return img

if __name__ == "__main__":
    # Exemplo de uso
    label_map = {
        0: '0',
        1: '5',
        2: '8',
        3: 'm',
        4: '2',
        5: '9',
        6: '4',
        7: '3',
        8: '7',
        9: '1',
        10: '6'
    }

    image_path = "data/image/sample_01.jpeg"
    txt_path = "data/label/sample_01.txt"

    img_with_annotations = visualize_image_annotations_universal(
        image_path,
        txt_path,
        label_map=label_map,
        draw_boxes=True,
        draw_segments=True,
        box_color=(255, 0, 0),
        segment_color=(0, 255, 0),
        line_width=2,
    )

    img_with_annotations.show()