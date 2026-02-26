"""
Пример детекции YOLO с визуализацией через OpenCV.
Рисует боксы, подписи классов и уверенность на изображении.
"""
import cv2
from ultralytics import YOLO

# Ширина по которой ресайзим с сохранением пропорций
TARGET_WIDTH = 640


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
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(
            frame, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA
        )

    return frame


def main():
    model = YOLO("yolo26n.pt")

    # Вариант 1: одна картинка
    image_path = r"src\img.jpg"
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Не удалось загрузить: {image_path}")
        return

    frame = resize_keep_aspect(frame)
    results = model(frame, verbose=False)
    frame = draw_detections(frame, results, model)

    cv2.imshow("YOLO + OpenCV", frame)
    print("Нажмите любую клавишу для выхода.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Вариант 2: видео (раскомментируйте для проверки на видео)
    # cap = cv2.VideoCapture(r"src\apples.mp4")
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frame = resize_keep_aspect(frame)
    #     results = model(frame, verbose=False)
    #     frame = draw_detections(frame, results, model)
    #     cv2.imshow("YOLO + OpenCV", frame)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
