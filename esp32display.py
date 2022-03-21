import cv2
from skimage import io
img_path = 'kohli.jpg'
xml_path = 'face_train.xml'
img = cv2.imread(img_path,0)   
face_detector = cv2.CascadeClassifier(xml_path)
faces = face_detector.detectMultiScale(img,scaleFactor = 1.1)
for face in faces:
    top_left_x= face[0]
    top_left_y= face[1]
    w= face[2]
    h= face[3]
    bottom_right_x= top_left_x+w
    bottom_right_y= top_left_y+h
    cv2.rectangle(img,
                  (top_left_x,top_left_y),(bottom_right_x,bottom_right_y),
                  (0,255,0),1)
cv2.imshow("Faces",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
