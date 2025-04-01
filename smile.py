import cv2
import mediapipe

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    flippedimg = cv2.flip(img, 1)
    cv2.imshow("Image", flippedimg)
    cv2.waitKey(1)