# -*- coding: utf-8 -*-

from keras import backend as K
import imutils
from keras.models import load_model
import numpy as np
import keras
import dlib
import cv2, os, sys
import face_recognition
import tensorflow as tf
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import pygame
import threading

# ----------- Khởi tạo pygame mixer (1 lần duy nhất) -----------
pygame.mixer.init()

# ----------- Class lấy chỉ số landmark mắt -----------
class FacialLandMarksPosition:
    left_eye_start_index, left_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    right_eye_start_index, right_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# ----------- Hàm dự đoán trạng thái mắt (mở hoặc nhắm) -----------
def predict_eye_state(model, image):
    if image is None or image.size == 0:
        return 1
    try:
        image = cv2.resize(image, (20, 10))
    except cv2.error:
        return 1
    image = image.astype(dtype=np.float32)
    image_batch = np.reshape(image, (1, 10, 20, 1))
    image_batch = keras.applications.mobilenet.preprocess_input(image_batch)
    try:
        prediction = model.predict(image_batch, verbose=0)
        return np.argmax(prediction[0])
    except:
        return 1

# ----------- Hàm phát âm thanh cảnh báo -----------
def play_alert_sound():
    if alert_sound_enabled:
        try:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.load('alert.mp3')
                pygame.mixer.music.play()
        except:
            pass

# ----------- Load model landmark dlib -----------
facial_landmarks_predictor = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(facial_landmarks_predictor):
    sys.exit(1)
try:
    predictor = dlib.shape_predictor(facial_landmarks_predictor)
except:
    sys.exit(1)

# ----------- Load model nhận diện trạng thái mắt -----------
model_path = 'weights.149-0.01.hdf5'
if not os.path.exists(model_path):
    sys.exit(1)

model = None
try:
    custom_objects = {'Adagrad': keras.optimizers.Adagrad(learning_rate=0.01)}
    model = load_model(model_path, custom_objects=custom_objects)
except:
    try:
        model = load_model(model_path, compile=False)
        model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.01),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    except:
        try:
            model = load_model(model_path)
        except:
            sys.exit(1)

if model is None:
    sys.exit(1)

# ----------- Mở webcam -----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit(1)

# ----------- Các biến khởi tạo -----------
scale = 0.5
countClose = 0
currState = 0
alarmThreshold = 4
alert_sound_enabled = True  # Mặc định bật âm thanh

# ----------- Vòng lặp xử lý video -----------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    image = frame.copy()
    original_height, original_width = image.shape[:2]
    resized_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    resized_height, resized_width = l.shape[:2]

    height_ratio = original_height / resized_height
    width_ratio = original_width / resized_width

    face_locations = face_recognition.face_locations(l, model='hog')

    if len(face_locations):
        top, right, bottom, left = face_locations[0]

        x1 = int(left * width_ratio)
        y1 = int(top * height_ratio)
        x2 = int(right * width_ratio)
        y2 = int(bottom * height_ratio)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dlib_rect = dlib.rectangle(x1, y1, x2, y2)

        try:
            shape = predictor(gray, dlib_rect)
            face_landmarks = face_utils.shape_to_np(shape)
        except:
            continue

        left_eye_indices = face_landmarks[FacialLandMarksPosition.left_eye_start_index:
                                          FacialLandMarksPosition.left_eye_end_index]
        right_eye_indices = face_landmarks[FacialLandMarksPosition.right_eye_start_index:
                                           FacialLandMarksPosition.right_eye_end_index]

        try:
            (x, y, w, h) = cv2.boundingRect(np.array([left_eye_indices]))
            left_eye = gray[y:y + h, x:x + w]
            left_eye_state = predict_eye_state(model=model, image=left_eye)
        except:
            left_eye_state = 1

        try:
            (x, y, w, h) = cv2.boundingRect(np.array([right_eye_indices]))
            right_eye = gray[y:y + h, x:x + w]
            right_eye_state = predict_eye_state(model=model, image=right_eye)
        except:
            right_eye_state = 1

        left_eye_open = 'yes' if left_eye_state == 1 else 'no'
        right_eye_open = 'yes' if right_eye_state == 1 else 'no'

        if left_eye_open == 'no' and right_eye_open == 'no':
            countClose += 1
            currState = 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            countClose = 0
            currState = 0
            if left_eye_open == 'yes' and right_eye_open == 'yes':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    else:
        countClose = 0
        currState = 0

    frame = cv2.flip(frame, 1)

    if countClose > alarmThreshold:
        cv2.putText(frame, "BUON NGU! CANH BAO!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        threading.Thread(target=play_alert_sound, daemon=True).start()

    # Hiển thị trạng thái bật/tắt âm thanh
    sound_status = "Sound: ON" if alert_sound_enabled else "Sound: OFF"
    cv2.putText(frame, sound_status, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Sleep Detection - Nhan Q de thoat | Bam M de bat/tat am thanh', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('m'):
        alert_sound_enabled = not alert_sound_enabled  # Bật/Tắt tiếng khi bấm M

# ----------- Giải phóng tài nguyên -----------
cap.release()
cv2.destroyAllWindows()
