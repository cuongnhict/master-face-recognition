import numpy as np
import cv2
import os
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow_addons as tfa

from model import base_network
from utils import save_pickle, load_pickle


img_dir = './data/image'
pkl_dir = './data/pkl'


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


# face_imgs, labels = load_images(img_dir)
# save_pickle(face_imgs, f'{pkl_dir}/faces.pkl')
# save_pickle(labels, f'{pkl_dir}/labels.pkl')


# region Embedding
# face_imgs = load_pickle(f'{pkl_dir}/faces.pkl')
# labels = load_pickle(f'{pkl_dir}/labels.pkl')


# def blob_img(img, out_size=(300, 300), scaleFactor=1.0, mean=(104.0, 177.0, 123.0)):
#     imageBlob = cv2.dnn.blobFromImage(img,
#                                       scalefactor=scaleFactor,
#                                       size=out_size,
#                                       mean=mean,
#                                       swapRB=False,
#                                       crop=False)
#     return imageBlob


# def embedding_faces(pretrain_path, face_imgs):
#     encoder = cv2.dnn.readNetFromTorch(pretrain_path)
#     emb_vecs = []
#     for face_img in face_imgs:
#         face_blob = blob_img(face_img,
#                              out_size=(96, 96),
#                              scaleFactor=1/255.0,
#                              mean=(0, 0, 0))
#         encoder.setInput(face_blob)
#         vec = encoder.forward()
#         emb_vecs.append(vec)
#     return emb_vecs


# pretrain_path = './weight/nn4.small2.v1.t7'
# embed_faces = embedding_faces(pretrain_path, face_imgs)
# save_pickle(embed_faces, f'{pkl_dir}/embed_blob_faces.pkl')
# endregion


face_imgs = load_pickle(f'{pkl_dir}/faces.pkl')
labels = load_pickle(f'{pkl_dir}/labels.pkl')


def preprocessing(face_imgs):
    face_imgs_resized = []
    for face_img in face_imgs:
        face_img_resized = cv2.resize(face_img, (224, 224))
        face_imgs_resized.append(face_img_resized)
    return face_imgs_resized


face_imgs_resized = preprocessing(face_imgs)
X = np.stack(face_imgs_resized)

ids = np.arange(len(labels))
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    labels,
                                                    test_size=0.2)

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

X_train_vec = model.predict(X_train)
X_test_vec = model.predict(X_test)
y_preds = []
for vec in X_test_vec:
    vec = vec.reshape(1, -1)
    y_pred = _most_similarity(X_train_vec, vec, y_train)
    y_preds.append(y_pred)

print(accuracy_score(y_preds, y_test))
