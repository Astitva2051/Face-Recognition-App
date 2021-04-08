import cv2
import numpy as np
import os

dir = r'/image'  #Base Directory
people = ['Person 1']   #list of people whose faces are to be detected

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(dir,person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv2.imread(img_path)
            gray = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
            features.append(gray)
            labels.append(label)

create_train()

#print(len(features))  "can be used to check if function is running correctly"
#print((len(labels)))  "can be used to check if function is running correctly"

features = np.array(features,dtype='object')     #convert list to array of numpy
labels = np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(np.asarray(features) , np.asarray(labels))    #it trains the model on provided images

face_recognizer.save('face_trained.yml')             #model is saved so that training is not needed again
np.save('features.npy',features)
np.save('labels.npy',labels)
print("training done...................")