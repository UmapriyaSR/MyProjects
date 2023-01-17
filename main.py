import cv2
import argparse

from PIL import Image
from utils import *

def putface(img, face, x, y, w, h):
    if(face == 'No Mask'):
        cv2.putText(img, face, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.rectangle(img, (x, y-5), (x+w, y+h), (0,0,255), 2)
    elif(face == 'Mask'):
        cv2.putText(img, face, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y-5), (x+w, y+h), (0,255,0), 2)

def predict_allCharacters(imgs):
    output = model(imgs)
    probs = torch.nn.functional.softmax(output, dim=1)
    conf, classes = torch.max(probs, 1)
    return conf

def predict(img_detect, model):
    img_detect = cv2.resize(img_detect, (32, 32)) #Resize 32x32
    img = Image.fromarray(img_detect)     
    
    img = data_transform(img)    
    img = img.view(1, 3, 32, 32) #View in tensor
    img = Variable(img)      
    
    model.eval() #Set eval mode

    #To Cuda
    model = model
    img = img

    output = model(img)
    probs = torch.nn.functional.softmax(output, dim=1)
    conf, classes = torch.max(probs, 1)
    print(conf*100)
    predicted = torch.argmax(output)
    p = label2id[predicted.item()]

    return  predicted

if __name__ == "__main__":
    #Define parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="Img or Video")
    parser.add_argument("--path", help="Link direct")
    opt = parser.parse_args()
    
    #Load model
    model = CNN()
    model = model
    model.load_state_dict(torch.load('weights/Face-Mask-Model.pt'))

    if(opt.mode == "Image"):
        #Load haarlike feature
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_default.xml')

        #Detect face
        img = cv2.imread(opt.path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            img2 = img[y+2:y+h-2, x+2:x+w-2]
            emo = predict(img2, model)  #face index 
            face = label2id[emo.item()]
            putface(img, face, x, y, w, h)

        cv2.imwrite("Result.jpg", img)
    else:
        pass
