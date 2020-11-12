import numpy as np
import cv2
import os
import time
from datetime import datetime
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

input_path = './data/video/mica-cam.mp4'
video_cap = cv2.VideoCapture(input_path)

output_path = './data/video/mica-cam-output.mp4'
out = cv2.VideoWriter(output_path,
                      cv2.VideoWriter_fourcc(*'DIVX'),
                      15,
                      (1920, 1080))
idx = 1

while(video_cap.isOpened()):
    print(idx)
    idx += 1
    ret, frame = video_cap.read()
    if not ret or idx > 100:
        break
    # Detect the faces
    t0 = datetime.now()
    print('t0 = ', t0)
    detections = detector.detect(frame)
    t1 = datetime.now()
    print('t1 = ', t1)

    t0 = datetime.now()
    print('t0 = ', t0)
    face_imgs = []
    for i, detect in enumerate(detections):
        x1, y1 = int(detect[0]), int(detect[1])
        x2, y2 = int(detect[2]), int(detect[3])
        face_img = frame[y1:y2, x1:x2].copy()
        face_imgs.append(face_img)
    if len(face_imgs) > 0:
        face_imgs_resized = preprocessing(face_imgs)
        face_imgs_resized = np.stack(face_imgs_resized)

        # Recognize the faces
        names = []
        vecs = recognitor.predict(face_imgs_resized)
        for vec in vecs:
            vec = vec.reshape(1, -1)
            name, pred_proba = most_similarity(X_train_vec, vec, y_train)
            name = None if name is None else f'{name}: {round(pred_proba * 100, 2)}%'
            names.append(name)

        for detect, face_img, name in zip(detections, face_imgs, names):
            if name is not None:
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
    # cv2.imshow('img', frame)
    # cv2.waitKey()
    out.write(frame)
    t1 = datetime.now()
    print('t1 = ', t1)

video_cap.release()
cv2.destroyAllWindows()
out.release()
