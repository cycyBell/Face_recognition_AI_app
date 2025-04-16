import cv2
from keras_facenet import FaceNet
import numpy as np
import numpy.linalg as lg


    
def get_embeddings(face):
    
    face = cv2.resize(face,(160,160))
    face = np.expand_dims(face, axis=0)
    embedder = FaceNet()
    image_emebddings = embedder.embeddings(face)
    return image_emebddings[0]
def cosine(u,v):
    return np.dot(u,v)/(lg.norm(u)*lg.norm(v))
