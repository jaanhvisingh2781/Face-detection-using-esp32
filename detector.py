import cv2
from skimage import io
#GLOBALS
img_path = 'kohli.jpg'
xml_path = 'face_train.xml'
face_detector = cv2.CascadeClassifier(xml_path)
url = 'http://192.168.125.164/capture' 

def detect():                     #function that detects faces in an image and draws a rectangle over there
    img = io.imread(url)
    faces = face_detector.detectMultiScale(img, scaleFactor=2)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    for face in faces:
        top_left_x = face[0]
        top_left_y = face[1]
        width = face[2]
        height = face[3]

        bottom_right_x = top_left_x + width
        bottom_right_y = top_left_y + height

        cv2.rectangle(img,                                      #image or the object
                        (top_left_x, top_left_y),               #top corner co-ordinates
                        (bottom_right_x, bottom_right_y),       #bottom corner co-ordinates
                        (255, 0, 0),                            #rgb value of rectgangle
                        2)                                      #width of the rectangle
    return img 

cv2.destroyAllWindows()