import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

# this time we wont be plotting connections since there are too many points to consider.
mpdraw = mp.solutions.drawing_utils
mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh() #This is needed to create objects.

# FaceMesh(), takes 4 parameters, static_image_mode(False,default), max_num_faces, min_detection_confidence, min_tracking_confidence.

while True:
    success, img = cap.read()
    flippedimg = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(flippedimg, cv2.COLOR_BGR2RGB)

    cv2.imshow("image", flippedimg)
    cv2.waitKey(1)