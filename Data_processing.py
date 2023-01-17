
import os #for os functions to join paths and files
import numpy as np
import cv2

# count = 0
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_default.xml')
#The repository has the pretrained classifier models stored in XML files, and can be read with the OpenCV methods
#Using haar-like features to crop face

def face_crop(path_link, filename):
    print(path_link)
    image = cv2.imread(path_link)

    #feature extraction
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #RGB to Grayscale conversion
    face_img = face_cascade.detectMultiScale(grayscale, 1.3, 5) #Detects objects of different sizes in the input image. 
   
    print(face_img)
    for (x,y,w,h) in face_img:
        image = image[y:y+h, x:x+w] #crop image for removing noise data
        os.remove(path_link) #remove old image 
        cv2.imwrite(filename, image) #save the image

for i in os.listdir('dataset/'):
    file = os.path.join('dataset/', i) #adding path to image
    file = file + '/'
    count = 0
    for v in os.listdir(file):    #getting processed image for training
        image = os.path.join(file, v)
        print(image)
        filename = str(count) + '_after' +'.jpg'
        filename = os.path.join(file, filename)
        face_crop(image, filename)
        count += 1
    # print(filename, count)

