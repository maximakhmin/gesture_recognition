import cv2
from scripts.model import GestureModel

gestureModel = GestureModel(buffer_size=60, interval=5)  

# cap = cv2.VideoCapture("./test/test1.mp4")
cap = cv2.VideoCapture("./test/test2.mp4")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    word = gestureModel.predict(image)
    if word!="":
        print(word)