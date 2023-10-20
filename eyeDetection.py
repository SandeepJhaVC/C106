import cv2

eye_casscade = cv2.CascadeClassifier("C:/Python312/Lib/site-packages/cv2/data/haarcascade_eye.xml")

eye = cv2.VideoCapture(0)

while True:

    ret, frame = eye.read()
    eyeGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eyeScale = eye_casscade.detectMultiScale(eyeGray,1.1,5)
    print(eyeScale)

    for (x,y,w,h) in eyeScale:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),5)

    cv2.imshow('frame',frame)
    if cv2.waitKey(25)==32:
        break
eye.release()
cv2.destroyAllWindows()