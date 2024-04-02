import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
import numpy as np
import numpy.linalg as lg

def collect_data():
    detector = MTCNN()
    vid = cv2.VideoCapture(0)
    faces =[]
    l = 0
    while(True): 
        
        # Capture the video frame 
        # by frame 
        ret, frame = vid.read() 
    
        # Display the resulting frame 
        cv2.imshow('Scanning your face', frame)
        detected = detector.detect_faces(frame)
        if len(detected) == 1 :
            x,y,w,h = detected[0]["box"]
            face = frame[y:y+h,x:x+w]
            faces.append(face)
            l +=1
    

        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if (cv2.waitKey(1) & 0xFF == ord('q')) | l ==6:
            break
    
    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 
    return faces

def get_embeddings(face):
    
    face = cv2.resize(face,(160,160))
    face = np.expand_dims(face, axis=0)
    embedder = FaceNet()
    image_emebddings = embedder.embeddings(face)
    return image_emebddings[0]
def cosine(u,v):
    return np.dot(u,v)/(lg.norm(u)*lg.norm(v))
