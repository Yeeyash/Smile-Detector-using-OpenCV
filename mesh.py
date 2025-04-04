import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# this time we wont be plotting connections since there are too many points to consider.
mpdraw = mp.solutions.drawing_utils
mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh(static_image_mode=False) #This is needed to create objects.
drawspec = mpdraw.DrawingSpec(color = (0,255,0),thickness=1, circle_radius=2)


currentlysimling = False
previouslysmiling = False




lasttime = 0
debouncetime = 500
smiled = 0

upperlistindices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lowerlipindices = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
innerupperlistindices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
innerlowerlipindices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 317, 14]




# FaceMesh(), takes 4 parameters, static_image_mode(False,default), max_num_faces, min_detection_confidence, min_tracking_confidence.

def issmiling(liplandmark):
    left_mouth_corner = facelms.landmark[61]
    right_mouth_corner = facelms.landmark[291]
    upper_lip_top = facelms.landmark[0]
    lower_lip_bottom = facelms.landmark[17]

    mouth_width = (abs(right_mouth_corner.x - left_mouth_corner.x))
    mouth_height = (abs(upper_lip_top.y - lower_lip_bottom.y))
            
    mouth_ratio = mouth_width/mouth_height if mouth_height != 0 else 0

    return mouth_ratio >2.5 and mouth_height*100 > 3.5

while True:
    success, img = cap.read()
    flippedimg = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(flippedimg, cv2.COLOR_BGR2RGB)




    results = facemesh.process(imgrgb)

    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks: #lms - landmarks
            # mpdraw.draw_landmarks(flippedimg, facelms, mpfacemesh.FACEMESH_CONTOURS, drawspec) #broke here.

            ih, iw, _ = img.shape

            liplms = {}
            alllipindx = upperlistindices + lowerlipindices + innerlowerlipindices + innerupperlistindices
            for idx in alllipindx:
                landmark = facelms.landmark[idx]
                x, y = int(landmark.x * iw), int(landmark.y * ih)

                liplms[idx] = (x,y)
                cv2.circle(flippedimg, (x,y), 2, (0,255,0), -1)
            
            if issmiling(liplms) != previouslysmiling:
                currenttime = time.time() * 1000 #time in miliseconds.
                print("smliked", currenttime, lasttime)
                if (currenttime - lasttime) > debouncetime:
                    print(smiled)
                    smiled = smiled + 1
                    lasttime = currenttime

            previouslysmiling = issmiling(liplms)

            # printing smiled(smliked) whenever detected, error in timing.

            # mpdraw.draw_landmarks(flippedimg, facelms, mpfacemesh.FACEMESH_CONTOURS, drawspec)


    cv2.imshow("image", flippedimg)
    cv2.waitKey(1)