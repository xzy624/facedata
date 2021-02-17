import cv2
faceCascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(r'haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
ok = True
while ok:
    ok, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(32, 32)
    )
    result=[]
    for (x, y, w, h) in faces:
        fac_gray = gray[y: (y+h), x: (x+w)]
        result = []
        eyes = eyeCascade.detectMultiScale(fac_gray, 1.3, 2)
        for (ex, ey, ew, eh) in eyes:
           result.append((x+ex, y+ey, ew, eh))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    for (ex, ey, ew, eh) in result:
        cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    cv2.imshow('video', img)
    k = cv2.waitKey(1)
    if k == 27:    # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()