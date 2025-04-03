import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

# this time we wont be plotting connections since there are too many points to consider.
mpdraw = mp.solutions.drawing_utils
mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh(static_image_mode=False) #This is needed to create objects.
drawspec = mpdraw.DrawingSpec(color = (0,255,0),thickness=1, circle_radius=2)

lipindices = list(range(61,68)) + list(range(78,95))

# FaceMesh(), takes 4 parameters, static_image_mode(False,default), max_num_faces, min_detection_confidence, min_tracking_confidence.

while True:
    success, img = cap.read()
    flippedimg = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(flippedimg, cv2.COLOR_BGR2RGB)

    results = facemesh.process(imgrgb)

    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks: #lms - landmarks
            # mpdraw.draw_landmarks(flippedimg, facelms, mpfacemesh.FACEMESH_CONTOURS, drawspec) #broke here.

            left_mouth_corner = facelms.landmark[61]
            right_mouth_corner = facelms.landmark[291]
            upper_lip_top = facelms.landmark[0]
            lower_lip_bottom = facelms.landmark[17]

            mouth_width = (abs(right_mouth_corner.x - left_mouth_corner.x))
            mouth_height = (abs(upper_lip_top.y - lower_lip_bottom.y))
            
            mouth_ratio = mouth_width/mouth_height if mouth_height != 0 else 0

            if mouth_ratio >2.5:
                cv2.putText(img, "smili", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
                print(mouth_ratio)
                print("smil")

            mpdraw.draw_landmarks(flippedimg, facelms, mpfacemesh.FACEMESH_CONTOURS, drawspec)


    cv2.imshow("image", flippedimg)
    cv2.waitKey(1)