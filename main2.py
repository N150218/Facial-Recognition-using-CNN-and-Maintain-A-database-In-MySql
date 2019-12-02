import numpy as np
import sys
import pygame, time
from gtts import gTTS 
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import mysql.connector
import sys
from PIL import Image
import PIL.Image
import time
db = mysql.connector.connect(user='root', password='',
                              host='localhost',
                              database='security')
np.set_printoptions(threshold=2**31-1)

face_cascade=cv2.CascadeClassifier('/home/sumanth/Desktop/Project/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
q=0
while(1>2):
    ret,frame=cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   # print(gray)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.6,minNeighbors=6)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        
        roi_color=frame[y:y+h,x:x+w]
        img_item= 'my-image.png'
        cv2.imwrite(img_item,roi_color)    
        if(len(faces)>0):
            q=1
            break
        color=(255,0,0)#bgr
        stroke=2
        end_cord_x=x+w
        end_cord_y=y+h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)

    cv2.imshow('frame',frame)
    if(q==1):
        break
    if cv2.waitKey(20) & 0xFF== ord('s'):
       break
    
cap.release()
cv2.destroyAllWindows()
#img=cv2.imread('img12.png',1)
img=cv2.imread('sumanth.png',1)
img1=cv2.resize(img,(96,96), interpolation = cv2.INTER_AREA)
cv2.imwrite('img1.png',img1)
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Total Params:", FRmodel.count_params())
print(FRmodel.summary)
def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis = -1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis = -1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha) 
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

    return loss
with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)
    
    print("loss = " + str(loss.eval()))
    
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)
database = {}
idatabase={"sumanth":1,"vara":2}
database["sumanth"] = img_to_encoding("images/sumanth.jpg", FRmodel)
#database["sumanth"].append()
#database["sumanth"].append(img_to_encoding("images/sumanth2.jpg", FRmodel))
#database["sumanth"].append(img_to_encoding("images/sumanth3.jpg", FRmodel))
#database["sumanth"].append(img_to_encoding("images/sumanth4.jpg", FRmodel))
#database["sumanth"].append(img_to_encoding("images/sumanth5.jpg", FRmodel))
database["vara"] = img_to_encoding("images/vara3.jpg", FRmodel)
def who_is_it(image_path, database, model,idatabase):
    
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    ## Step 2: Find the closest encoding ##
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding-db_enc)
        print(dist," ",name,"\n")
        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 2.0:
        pygame.init()
        pygame.mixer.music.load('not.mp3')
        pygame.mixer.music.play()
        time.sleep(5)
        pygame.mixer.music.fadeout(5)
        print("Not in the database.")
        change=0
    else:
        mytext = "it's " + str(identity)
        language ='en'
        myobj = gTTS(text=mytext, lang=language, slow=False)
        myobj.save(str(identity)+".mp3")
        pygame.init()
        pygame.mixer.music.load(str(identity)+".mp3")
        pygame.mixer.music.play()
        time.sleep(5)
        pygame.mixer.music.fadeout(5)
        print ("it's" + str(identity) + ", the distance is " + str(min_dist))
        change=idatabase[str(identity)]
    return min_dist, identity,change
Values=who_is_it("img1.png", database, FRmodel,idatabase)
id=Values[2]
ti=str(time.ctime())
#image = Image.open('C:\Users\Abhi\Desktop\cbir-p\images.jpg')
blob_value = open('img1.png', 'rb').read()
#sql = 'INSERT INTO persons(name,in_1,out_1,photo,status) VALUES(%s,%s,%s,%s,%s)'    
#sql='SELECT status,in_1,out_1 from persons where id = 1 '
#args = ("teja",ti,ti,blob_value,0)
cursor=db.cursor()
#cursor.execute("CREATE TABLE `persons` ( `id` INT NOT NULL AUTO_INCREMENT, `name` TEXT NOT NULL ,`in_1` TEXT ,`out_1` TEXT  , `photo` BLOB NOT NULL ,`status` BOOLEAN NOT NULL DEFAULT false, PRIMARY KEY (`id`))")  
#cursor.execute(sql,args)
if(id==0):
    sql = 'INSERT INTO persons(name,in_1,photo,status) VALUES(%s,%s,%s,%s)'
    args = ("unknown",ti,blob_value,1)
    cursor.execute(sql,args)
else:
    sql='SELECT status,in_1,out_1 from persons where id = {}'.format(id)
    cursor.execute(sql)
    m=cursor.fetchall()
    for x in m:
        print(x)
    if(x[0]==0):
        ti1=ti+x[1]
        sql2 = "UPDATE persons SET in_1 = '{}',status='1' WHERE id={}".format(ti1,id)
        cursor.execute(sql2)
    else:
        ti1=ti+x[2]
        sql2 = "UPDATE persons SET out_1 = '{}',status='0' WHERE id={}".format(ti1,id)
        cursor.execute(sql2)
db.commit()
db.close()






