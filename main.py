"""
Пример детекции YOLO с визуализацией через OpenCV.
Рисует боксы, подписи классов и уверенность на изображении.
"""
import os
import sys
import json
import cv2
from ultralytics import YOLO

# Ширина по которой ресайзим с сохранением пропорций
TARGET_WIDTH = 640


def get_base_dir():
    """Папка с exe при сборке PyInstaller, иначе — папка скрипта."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resize_keep_aspect(frame, width=TARGET_WIDTH):
    """Ресайз кадра с сохранением пропорций; ширина = width."""
    h, w = frame.shape[:2]
    if w == width:
        return frame
    new_h = int(h * width / w)
    return cv2.resize(frame, (width, new_h))


def draw_detections(frame, results, model, conf_threshold=0.25):
    """
    Рисует боксы и подписи на кадре по результатам YOLO.
    """
    boxes = results[0].boxes
    names = model.names  # словарь id -> имя класса

    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"{names[cls_id]} {conf:.2f}"
        color = (0, 255, 0)  # BGR — зелёный
        thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y2 - th - 10), (x1 + tw, y2), color, -1)
        cv2.putText(
            frame, label, (x1, y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
        )

    return frame


def image_detection(image_path):
    base = get_base_dir()
    model = YOLO(os.path.join(base, "yolo26n.pt"))
    print(model.names)  # Словарь с классами (id: название)
    for _name in (list(model.names.values())):  # Список названий
        print(_name)
    # Вариант 1: одна картинка
    # image_path = r"src\minion.jpg"
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Не удалось загрузить: {image_path}")
        return

    frame = resize_keep_aspect(frame)
    results = model(frame, verbose=False)
    frame = draw_detections(frame, results, model)

    cv2.imshow("YOLO + OpenCV", frame)
    print("Нажмите любую клавишу для выхода.")
    cv2.waitKey(100000)
    cv2.destroyAllWindows()

def video_detection(image_path):
    base = get_base_dir()
    model = YOLO(os.path.join(base, "yolo26n.pt"))
    cap = cv2.VideoCapture(image_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_keep_aspect(frame)
        results = model(frame, verbose=False)
        frame = draw_detections(frame, results, model)
        cv2.imshow("YOLO + OpenCV", frame)
        key = cv2.waitKey(10)
        if key == ord("q") or key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    base = get_base_dir()
    image_path = os.path.join(base, "src", "apples.mp4")
    conf_path = os.path.join(base, "conf.json")
    with open(conf_path, "r", encoding="utf-8") as f:
        name_str = f.read()
        name_dict = json.loads(name_str)
        image_path = name_dict["name"]
    
    for ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}:
        if image_path.find(ext) > 0:
            print("image", ext)
            image_detection(image_path)
    
    for ext in {'.mp4', '.avi', '.mov', '.mkv'}:
        if image_path.find(ext) > 0:
            print("video", ext)
            video_detection(image_path)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
    
    
