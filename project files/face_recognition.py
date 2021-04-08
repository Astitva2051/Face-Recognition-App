import numpy as np
import cv2

haar_cascade = cv2.CascadeClassifier('haar_face.xml')

features = np.load('features.npy',allow_pickle=True)
labels = np.load('labels.npy',allow_pickle=True)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')                 #it reads the trained model.

people = ['Person 1']

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    for (x,y,w,h) in faces:
        faces_roi = gray[y:y+h,x:x+w]
        label, accuracy = face_recognizer.predict(faces_roi)  #predicts the person
        accuracy = int(accuracy)%100
        cv2.putText(frame,str(people[label]),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.putText(frame, "Accuracy : "+str(accuracy)+"%", (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('detected image', frame)
    if cv2.waitKey(1)==13 :   #breaks loop when enter is pressed
        break

cap.release()
cv2.destroyAllWindows()