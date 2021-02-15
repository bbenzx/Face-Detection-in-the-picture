import cv2

# Load the cascade, algorithmn of the program
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# Read the input image
img = cv2.imread('benz.jpg')
# Convert into grayscale before get into algorithmn
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(grayscale)
#loop for point all of face in picture
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

    roi_gray = grayscale[y:y+h, x:x+w]
    cv2.imshow('roi_gray', roi_gray)
    roi_color = img[y:y+h, x:x+w]
    cv2.imshow('roi_color', roi_color)
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
# Display the output
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()