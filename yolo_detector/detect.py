import cv2
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt

INPUT_DIR = os.path.abspath("data/raw_images")
OUTPUT_DIR = os.path.abspath("data/detected_phones")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Usar un modelo más preciso
# model = YOLO("yolov8m.pt")  # También puedes probar "yolov8l.pt"
MODEL_PATH = os.path.abspath("models/yolov8m.pt")
model = YOLO(MODEL_PATH)

PHONE_CLASS_ID = 67  # 'cell phone' en COCO

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(path)

    if img is None:
        print(f"No se pudo leer la imagen {filename}")
        continue

    height, width = img.shape[:2]
    print(f"Procesando {filename} - tamaño: {width}x{height}")

    # Umbral de confianza más bajo para permitir más detecciones
    results = model(img, conf=0.20)[0]

    phone_boxes = [box for box in results.boxes if int(box.cls.item()) == PHONE_CLASS_ID]
    print(f"Total detecciones: {len(results.boxes)} | Teléfonos detectados: {len(phone_boxes)}")

    phone_count = 0

    for box in phone_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        w, h = x2 - x1, y2 - y1
        if w < 10 or h < 10:
            print(f"Box descartado por tamaño pequeño: {w}x{h}")
            continue

        phone_count += 1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        text = f"Celular {phone_count}"
        # cv2.putText(img, label, (x1, max(y1 - 20, 0)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 7.0, (0, 255, 0), 10)
        font_scale = max(0.5, img.shape[0] / 1000.0)  # por ejemplo: 1080px -> 1.08
        thickness = max(1, int(font_scale * 2))

        text_x = x1 + 5
        text_y = y1 + int(30 * font_scale)

        # Fondo para mayor legibilidad
        cv2.putText(img, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)

        # Texto principal
        cv2.putText(img, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, lineType=cv2.LINE_AA)

    if phone_count == 0:
        print(f"No se detectaron teléfonos en {filename}")
    else:
        print(f"Detectados {phone_count} teléfonos en {filename}")

    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, img)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Detección - {filename}", fontsize=16)
    plt.axis('off')
    plt.show()
