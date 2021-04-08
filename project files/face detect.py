import  numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haar_face.xml')

cap = cv2.VideoCapture(0)  #to start record images from web cam.
count = 0               #it records the total no of images detected.

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    for (x,y,w,h) in faces:
        count=count+1
        roi_gray = gray[y:y+h, x:x+w]   #we need to crop the image to get only detected face
         # roi_color = frame[y:y + h, x:x + w]

        img_item = r'C:\Users\hp\PycharmProjects\learnopencv\image\person1\%d.jpg'%count  #DIR path to store images

        cv2.imwrite(img_item,roi_gray)    #it will store the images in the specified directory

        cv2.putText(frame,str(count),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1)==13 or count==100:   #loop will break when count=100 or enter is pressed.
        break

cap.release()  #never forget to put this statement otherwise web cam will not stop
cv2.destroyAllWindows()
print("dataset completed")



