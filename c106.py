import cv2

face_casscade = cv2.CascadeClassifier("C:/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)

while True:
    ret,frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = face_casscade.detectMultiScale(gray,1.1,5)
    print(face)

    for(x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),5)

    cv2.imshow('frame',frame)
    if cv2.waitKey(25)==32:
        break

video.release()
cv2.destroyAllWindows()