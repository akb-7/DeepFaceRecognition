import cv2
import numpy as np
import os



face_classifier = cv2.CascadeClassifier('./utils/haarcascade_frontalface_default.xml')

name = input("Enter the Person's Name ....")

def face_extractor(img):
    # Function detects faces and returns the cropped face
    
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    
    if faces == ():
        return None
    
    # Cropping the images    
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50, x:x+w+50]

    return cropped_face


cap = cv2.VideoCapture(-1)
count = 0 # for convenience for us to save the images so that it doesn't save the same file

while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400))

        # Save file in specified directory with unique name
        file_name_path = './Dataset/' + name +"-"+ str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        #print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")