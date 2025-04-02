import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

# this time we wont be plotting connections since there are too many points to consider.
mpdraw = mp.solutions.drawing_utils
mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh(static_image_mode=False) #This is needed to create objects.
drawspec = mpdraw.DrawingSpec(color = (0,255,0),thickness=1, circle_radius=2)


# FaceMesh(), takes 4 parameters, static_image_mode(False,default), max_num_faces, min_detection_confidence, min_tracking_confidence.

while True:
    success, img = cap.read()
    flippedimg = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(flippedimg, cv2.COLOR_BGR2RGB)

    results = facemesh.process(imgrgb)

    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks: #lms - landmarks
            mpdraw.draw_landmarks(flippedimg, facelms, mpfacemesh.FACEMESH_CONTOURS, drawspec) #broke here.

    cv2.imshow("image", flippedimg)
    cv2.waitKey(1)