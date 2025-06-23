import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from classifier.classify import classify_subclass  # asumimos que esto sí queda separado

# Cargar modelo YOLO aquí mismo (independiente)
# model = YOLO("yolov8m.pt")
MODEL_PATH = os.path.abspath("models/yolov8m.pt")
model = YOLO(MODEL_PATH)

PHONE_CLASS_ID = 67  # 'cell phone' en COCO dataset

def run_pipeline_on_image(img):
    height, width = img.shape[:2]
    results = model(img, conf=0.20)[0]

    phone_boxes = []
    for box in results.boxes:
        if int(box.cls.item()) == PHONE_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            phone_boxes.append((x1, y1, x2, y2))

    phone_count = 0
    for (x1, y1, x2, y2) in phone_boxes:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        w, h = x2 - x1, y2 - y1
        if w < 10 or h < 10:
            continue

        phone_count += 1
        crop_img = img[y1:y2, x1:x2]
        label, confidence = classify_subclass(crop_img)
        text = f"{label} ({confidence:.2f})"

        print(f"Predicción: {text}") 

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # cv2.putText(img, text, (x1 + 5, y1 + 35),
        #     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, lineType=cv2.LINE_AA)
        # cv2.putText(img, text, (x1 + 5, y1 + 35),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # Ajustar tamaño de fuente dinámicamente según altura de imagen
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

    return img, phone_count

def run_pipeline_on_folder(input_dir, output_dir):
        # Limpiar carpeta de salida (borrar archivos)
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # borrar archivo o link simbólico
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # borrar carpeta recursivamente
            except Exception as e:
                print(f"Error al borrar {file_path}: {e}")
    else:
        os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(input_dir, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"No se pudo leer la imagen {filename}")
            continue

        print(f"====================================================")
        print(f"Procesando {filename}")
        img_out, count = run_pipeline_on_image(img)
        if count == 0:
            print("No se detectaron teléfonos.")
        else:
            print(f"Detectados {count} teléfonos.")

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img_out)

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        plt.title(f"Detección y Clasificación - {filename}", fontsize=16)
        plt.axis('off')
        plt.show()

        # dpi = 100  # Dots per inch (ajustable)
        # h, w = img_out.shape[:2]
        # plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        # plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        # plt.title(f"Detección y Clasificación - {filename}", fontsize=14)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.abspath(os.path.join(script_dir, "data/raw_images"))
    output_folder = os.path.abspath(os.path.join(script_dir, "data/detected_phones"))
    run_pipeline_on_folder(input_folder, output_folder)
