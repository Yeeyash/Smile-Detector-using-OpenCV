import cv2
# primary camera was "0" if you want any other source, change the index inside "VideoCapture".
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1)