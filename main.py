"""
Пример детекции YOLO с визуализацией через OpenCV.
Рисует боксы, подписи классов и уверенность на изображении.
"""

import os
import sys
import json
import cv2
import time
from ultralytics import YOLO

# Ширина по которой ресайзим с сохранением пропорций
TARGET_WIDTH = 640

# MediaPipe будет инициализирован позже
hand_landmarker = None
pose_landmarker = None
face_landmarker = None
mp_drawing = None
drawing_styles = None
ImageFormat = None

MODEL_PATHS = {
    "hand": "mediapipe/models/hand_landmarker.task",
    "pose": "mediapipe/models/pose_landmarker_lite.task",
    "face": "mediapipe/models/face_landmarker.task",
}

MODEL_URLS = {
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def download_model(url, dest):
    import os

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        import urllib.request

        print(f"Downloading {dest}...")
        urllib.request.urlretrieve(url, dest)
        print(f"Downloaded {dest}")


def init_mediapipe():
    global \
        hand_landmarker, \
        pose_landmarker, \
        face_landmarker, \
        mp_drawing, \
        drawing_styles, \
        ImageFormat
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe import ImageFormat as IF

    ImageFormat = IF

    # Download models if needed
    download_model(MODEL_URLS["hand"], MODEL_PATHS["hand"])
    download_model(MODEL_URLS["pose"], MODEL_PATHS["pose"])
    download_model(MODEL_URLS["face"], MODEL_PATHS["face"])

    base_options = python.BaseOptions(model_asset_path=MODEL_PATHS["hand"])
    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2, running_mode=vision.RunningMode.IMAGE
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATHS["pose"])
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATHS["face"])
    face_options = vision.FaceLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)


def recognize_gesture(hand_landmarks):
    """Распознавание базовых жестов по landmarks руки"""
    if not hand_landmarks or len(hand_landmarks) < 21:
        return None

    landmarks = hand_landmarks

    # Tip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    thumb_ip = landmarks[3]
    index_pip = landmarks[6]
    middle_pip = landmarks[10]
    ring_pip = landmarks[14]
    pinky_pip = landmarks[18]

    # Open palm - все пальцы подняты
    if (
        index_tip.y < index_pip.y
        and middle_tip.y < middle_pip.y
        and ring_tip.y < ring_pip.y
        and pinky_tip.y < pinky_pip.y
    ):
        return "Open"

    # Fist - все пальцы согнуты
    if (
        index_tip.y > index_pip.y
        and middle_tip.y > middle_pip.y
        and ring_tip.y > ring_pip.y
        and pinky_tip.y > pinky_pip.y
    ):
        return "Fist"

    # Peace sign - V
    if (
        index_tip.y < index_pip.y
        and middle_tip.y < middle_pip.y
        and ring_tip.y > ring_pip.y
        and pinky_tip.y > pinky_pip.y
    ):
        return "Peace"

    # Thumbs up
    if thumb_tip.y < thumb_ip.y and index_tip.y > index_pip.y:
        return "Thumbs Up"

    # Thumbs down
    if thumb_tip.y > thumb_ip.y and index_tip.y > index_pip.y:
        return "Thumbs Down"

    # Like (OK sign) - большой и указательный образуют кольцо
    dx = abs(thumb_tip.x - index_tip.x)
    dy = abs(thumb_tip.y - index_tip.y)
    if dx < 0.05 and dy < 0.05:
        return "OK"

    # Pointing up
    if index_tip.y < index_pip.y and middle_tip.y > middle_pip.y:
        return "Point Up"

    # Pointing right/left
    if index_tip.y > index_pip.y and index_tip.x < landmarks[5].x:
        return "Point Right"
    if index_tip.y > index_pip.y and index_tip.x > landmarks[5].x:
        return "Point Left"

    return None


def calculate_average_fps(current_fps, fps_buffer):
    fps_buffer.pop(0)
    fps_buffer.append(current_fps)
    average_fps = sum(fps_buffer) / len(fps_buffer)
    return average_fps, fps_buffer


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

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # cv2.rectangle(frame, (x1, y2 - th - 10), (x1 + tw, y2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1, y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def image_detection(image_path):
    init_mediapipe()
    base = get_base_dir()
    model = YOLO(os.path.join(base, "yolo26n.pt"))
    print(model.names)  # Словарь с классами (id: название)
    for _name in list(model.names.values()):  # Список названий
        print(_name)
    # Вариант 1: одна картинка
    # image_path = r"src\minion.jpg"
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Не удалось загрузить: {image_path}")
        return

    original_frame = frame.copy()
    frame = resize_keep_aspect(frame)
    results = model(frame, verbose=False)
    frame = draw_detections(frame, results, model)

    # MediaPipe Detection
    from mediapipe import Image

    mp_image = Image(
        image_format=ImageFormat.SRGB,
        data=cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB),
    )

    hand_results = hand_landmarker.detect(mp_image)
    pose_results = pose_landmarker.detect(mp_image)

    # Draw hand landmarks
    if hand_results.hand_landmarks:
        for hand_idx, hand in enumerate(hand_results.hand_landmarks):
            color = (255, 0, 0) if hand_idx == 0 else (0, 255, 0)
            for landmark in hand:
                x, y = (
                    int(landmark.x * frame.shape[1]),
                    int(landmark.y * frame.shape[0]),
                )
                cv2.circle(frame, (x, y), 2, color, -1)
            # Draw connections
            HAND_CONNECTIONS = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (5, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (9, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (13, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (0, 17),
                (17, 18),
            ]
            for start, end in HAND_CONNECTIONS:
                if start < len(hand) and end < len(hand):
                    x1, y1 = (
                        int(hand[start].x * frame.shape[1]),
                        int(hand[start].y * frame.shape[0]),
                    )
                    x2, y2 = (
                        int(hand[end].x * frame.shape[1]),
                        int(hand[end].y * frame.shape[0]),
                    )
                    cv2.line(frame, (x1, y1), (x2, y2), color, 1)

    # Draw pose landmarks
    if pose_results.pose_landmarks:
        for landmark in pose_results.pose_landmarks[0]:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

    cv2.imshow("YOLO + MediaPipe Holistic", frame)
    print("Нажмите любую клавишу для выхода.")
    cv2.waitKey(100000)
    cv2.destroyAllWindows()


def video_detection(image_path):
    init_mediapipe()
    base = get_base_dir()
    model = YOLO(os.path.join(base, "yolo26n.pt"))
    cap = cv2.VideoCapture(image_path)

    # FPS calculation variables
    previousTime = 0
    N = 15
    fps_buffer = [0] * N

    # Drawing styles for hands
    HAND_CONNECTIONS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 17),
        (17, 18),
    ]
    POSE_CONNECTIONS = [
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (15, 17),
        (15, 19),
        (15, 21),
        (16, 18),
        (16, 20),
        (16, 22),
        (17, 18),
        (18, 20),
        (20, 22),
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame.copy()
        frame = resize_keep_aspect(frame)

        # YOLO Detection
        results = model(frame, verbose=False)
        frame = draw_detections(frame, results, model)

        # MediaPipe Detection
        from mediapipe import Image

        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB),
        )

        hand_results = hand_landmarker.detect(mp_image)
        pose_results = pose_landmarker.detect(mp_image)

        # Draw hand landmarks with skeleton
        gesture_text = ""
        if hand_results.hand_landmarks:
            for hand_idx, hand in enumerate(hand_results.hand_landmarks):
                color = (255, 0, 0) if hand_idx == 0 else (0, 255, 0)

                # Draw landmarks
                for landmark in hand:
                    x, y = (
                        int(landmark.x * frame.shape[1]),
                        int(landmark.y * frame.shape[0]),
                    )
                    cv2.circle(frame, (x, y), 2, color, -1)

                # Draw connections
                for start, end in HAND_CONNECTIONS:
                    if start < len(hand) and end < len(hand):
                        x1, y1 = (
                            int(hand[start].x * frame.shape[1]),
                            int(hand[start].y * frame.shape[0]),
                        )
                        x2, y2 = (
                            int(hand[end].x * frame.shape[1]),
                            int(hand[end].y * frame.shape[0]),
                        )
                        cv2.line(frame, (x1, y1), (x2, y2), color, 1)

                # Simple gesture recognition
                gesture = recognize_gesture(hand)
                if gesture:
                    # Draw gesture label
                    x = int(hand[0].x * frame.shape[1])
                    y = int(hand[0].y * frame.shape[0]) - 30
                    cv2.putText(
                        frame, gesture, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                    )
                    gesture_text = gesture

        # Draw pose landmarks
        if pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks[0]:
                x, y = (
                    int(landmark.x * frame.shape[1]),
                    int(landmark.y * frame.shape[0]),
                )
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

            # Draw pose connections
            for start, end in POSE_CONNECTIONS:
                if start < len(pose_results.pose_landmarks[0]) and end < len(
                    pose_results.pose_landmarks[0]
                ):
                    lm1 = pose_results.pose_landmarks[0][start]
                    lm2 = pose_results.pose_landmarks[0][end]
                    x1, y1 = int(lm1.x * frame.shape[1]), int(lm1.y * frame.shape[0])
                    x2, y2 = int(lm2.x * frame.shape[1]), int(lm2.y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # Calculate and display FPS
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime) if previousTime > 0 else 0
        previousTime = currentTime
        fps, fps_buffer = calculate_average_fps(fps, fps_buffer)

        cv2.putText(
            frame,
            str(int(fps)) + " FPS",
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (255, 100, 0),
            2,
        )

        cv2.imshow("YOLO + MediaPipe Holistic", frame)
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

    if type(image_path) is int:
        print("camera", image_path)
        video_detection(image_path)
        return

    for ext in {".mp4", ".avi", ".mov", ".mkv"}:
        if image_path.find(ext) > 0:
            print("video", ext)
            video_detection(image_path)

    for ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp"}:
        if image_path.find(ext) > 0:
            print("image", ext)
            image_detection(image_path)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
