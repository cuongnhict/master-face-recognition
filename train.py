import numpy as np
import cv2
import os
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from model import base_network
from utils import save_pickle, load_pickle, preprocessing, most_similarity


LOAD_IMAGES_TO_PICKLE = False
RETRAIN_MODEL = False
SAVE_TRAIN_VEC = False

img_dir = './data/image'
pkl_dir = './data/pkl'

face_imgs, labels = None, None

if LOAD_IMAGES_TO_PICKLE:
    def load_images(img_dir):
        face_imgs = []
        labels = []
        for label_dir in glob(f'{img_dir}/*'):
            label = os.path.basename(label_dir)
            for img_path in glob(f'{label_dir}/*'):
                face_img = cv2.imread(img_path)
                face_imgs.append(face_img)
                labels.append(label)
        return face_imgs, labels

    face_imgs, labels = load_images(img_dir)
    save_pickle(face_imgs, f'{pkl_dir}/faces.pkl')
    save_pickle(labels, f'{pkl_dir}/labels.pkl')
else:
    face_imgs = load_pickle(f'{pkl_dir}/faces.pkl')
    labels = load_pickle(f'{pkl_dir}/labels.pkl')

face_imgs_resized = preprocessing(face_imgs)
X = np.stack(face_imgs_resized)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    labels,
                                                    test_size=0.2)

model = None
if RETRAIN_MODEL:
    model = base_network()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tfa.losses.TripletSemiHardLoss())
    gen_train = tf.data.Dataset \
        .from_tensor_slices((X_train, y_train)) \
        .repeat() \
        .shuffle(1024) \
        .batch(32)
    history = model.fit(gen_train,
                        steps_per_epoch=50,
                        epochs=10)
    model.save('./model/model_triplot.h5')
else:
    model = keras.models.load_model('./model/model_triplot.h5')

X_train_vec = model.predict(X_train)
X_test_vec = model.predict(X_test)
y_preds = []
for vec in X_test_vec:
    vec = vec.reshape(1, -1)
    y_pred = most_similarity(X_train_vec, vec, y_train)
    y_preds.append(y_pred[0])

if SAVE_TRAIN_VEC:
    save_pickle(X_train_vec, f'{pkl_dir}/x_train_vec.pkl')
    save_pickle(y_train, f'{pkl_dir}/y_train.pkl')

print(len(y_preds))
print(len(y_test))
y_preds = list(map(lambda l: l if l is not None else 'unknown', y_preds))
print(accuracy_score(y_preds, y_test))
