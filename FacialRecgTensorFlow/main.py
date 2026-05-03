import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')


#for directory in os.listdir('lfw'):
 #   for file in os.listdir(os.path.join('lfw', directory)):
  ##     NEW_PATH = os.path.join(NEG_PATH, file)
    #    os.replace(EX_PATH, NEW_PATH)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    cv2.imshow('image collection', frame)
    
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()