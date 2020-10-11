import pickle
import cv2
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def preprocessing(face_imgs):
    face_imgs_resized = []
    for face_img in face_imgs:
        face_img_resized = cv2.resize(face_img, (224, 224))
        face_imgs_resized.append(face_img_resized)
    return face_imgs_resized

def most_similarity(embed_vecs, vec, labels):
    sim = cosine_similarity(embed_vecs, vec)
    sim = np.squeeze(sim, axis=1)
    argmax = np.argsort(sim)[::-1][:1]
    label = [labels[idx] for idx in argmax][0]
    return label
