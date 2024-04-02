import keras
from keras_facenet import FaceNet
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from mtcnn import MTCNN


from datacollection import cosine



def detect_face(image):
    detector = MTCNN()
    detected = detector.detect_faces(image)
    if detected:
      x,y,w,h = detected[0]['box']
      face = image[y:y+h,x:x+w]
    else:
      return detected
    return face



def get_embeddings(face):
  face = cv2.resize(face,(160,160))
  face = np.expand_dims(face, axis=0)
  embedder = FaceNet()
  image_emebddings = embedder.embeddings(face)
  return image_emebddings[0]




def get_image_embed_frames(parent_dir):
  embed_table = []
  celebrities = []
  i = 0
  for celebrity in os.listdir(parent_dir):
    path_to_image = os.path.join(parent_dir, celebrity)
    paths_to_images = os.listdir(path_to_image)
    count = 0
    for image in paths_to_images:
      if count == 15:
        break
      image = cv2.cvtColor(cv2.imread(os.path.join(path_to_image,image)), cv2.COLOR_BGR2RGB)
      face = detect_face(image)
      if len(face) != 0:
        embedding = get_embeddings(face)
        embed_table.append(embedding)
        count += 1
    celebrities.extend([celebrity]*15)
  image_set = pd.DataFrame(embed_table)
  image_set["Name"] = celebrities
  return image_set



def ismatch(im_embed1, im_embed2):
  return cosine(im_embed1,im_embed2)>=0.5