import numpy as np
import cv2
import os
from glob import glob
import face_detection

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from model import base_network
from utils import save_pickle, load_pickle, preprocessing, most_similarity


detector = face_detection.build_detector('RetinaNetMobileNetV1',
                                         confidence_threshold=.5,
                                         nms_iou_threshold=.3)
recognitor = keras.models.load_model('./model/model_triplot.h5')

pkl_dir = './data/pkl'
X_train_vec = load_pickle(f'{pkl_dir}/x_train_vec.pkl')
y_train = load_pickle(f'{pkl_dir}/y_train.pkl')

file_path = './data/video/mica-cam.mp4'
video_cap = cv2.VideoCapture(file_path)
while(video_cap.isOpened()):
    ret, frame = video_cap.read()
    if not ret:
        break
    # Detect the faces
    detections = detector.detect(frame)
    face_imgs = []
    for i, detect in enumerate(detections):
        x1, y1 = int(detect[0]), int(detect[1])
        x2, y2 = int(detect[2]), int(detect[3])
        face_img = frame[y1:y2, x1:x2].copy()
        face_imgs.append(face_img)
    face_imgs_resized = preprocessing(face_imgs)
    face_imgs_resized = np.stack(face_imgs_resized)

    # Recognize the faces
    names = []
    vecs = recognitor.predict(face_imgs_resized)
    for vec in vecs:
        vec = vec.reshape(1, -1)
        name = most_similarity(X_train_vec, vec, y_train)
        names.append(name)

    for detect, face_img, name in zip(detections, face_imgs, names):
        x1, y1 = int(detect[0]), int(detect[1])
        x2, y2 = int(detect[2]), int(detect[3])

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,
                    name,
                    (x1, y2+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)

    frame = cv2.resize(frame,
                       (int(frame.shape[1]/1.5), int(frame.shape[0]/1.5)))
    cv2.imshow('frame', frame)
    cv2.waitKey()

video_cap.release()
cv2.destroyAllWindows()
