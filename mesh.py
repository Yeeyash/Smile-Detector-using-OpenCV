import cv2
import mediapipe as mp
import time

currentlysimling = False
previouslysmiling = False

lasttime = 0
debouncetime = 1000
smiled = 0

upperlistindices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lowerlipindices = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
innerupperlistindices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
innerlowerlipindices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 317, 14]

class Smile_Detector():
    def __init__(self, staticMode=False, maxFaces=2, DetectionCon=0.5, TrackCon=0.5):
        
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.DetectionCon = DetectionCon
        self.TrackCon = TrackCon

        # this time we wont be plotting connections since there are too many points to consider.
        self.mpdraw = mp.solutions.drawing_utils
        self.mpfacemesh = mp.solutions.face_mesh
        self.facemesh = self.mpfacemesh.FaceMesh(static_image_mode = self.staticMode, max_num_faces = self.maxFaces, min_detection_confidence = self.DetectionCon, min_tracking_confidence = self.TrackCon)
        self.drawspec = self.mpdraw.DrawingSpec(color = (0,255,0),thickness=1, circle_radius=2)

    def lipemesh(self, img, draw= True):
        self.imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.facemesh.process(self.imgrgb)

        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks: #lms - landmarks
                self.mpdraw.draw_landmarks(img, facelms, self.mpfacemesh.FACEMESH_CONTOURS, self.drawspec, self.drawspec) #broke here.

                # ih, iw, _ = self.img.shape

                # liplms = {}
                # alllipindx = upperlistindices + lowerlipindices + innerlowerlipindices + innerupperlistindices
                # for idx in alllipindx:
                #     landmark = facelms.landmark[idx]
                #     x, y = int(landmark.x * iw), int(landmark.y * ih)

                #     liplms[idx] = (x,y)
                #     cv2.circle(img, (x,y), 2, (0,255,0), -1)
            
        return img
   
# FaceMesh(), takes 4 parameters, static_image_mode(False,default), max_num_faces, min_detection_confidence, min_tracking_confidence.

# def issmiling(liplandmark):
#     left_mouth_corner = facelms.landmark[61]
#     right_mouth_corner = facelms.landmark[291]
#     upper_lip_top = facelms.landmark[0]
#     lower_lip_bottom = facelms.landmark[17]

#     mouth_width = (abs(right_mouth_corner.x - left_mouth_corner.x))
#     mouth_height = (abs(upper_lip_top.y - lower_lip_bottom.y))
            
#     mouth_ratio = mouth_width/mouth_height if mouth_height != 0 else 0

#     return mouth_ratio >2.5 and mouth_height*100 > 3.5

    # if issmiling(liplms) != previouslysmiling:
    #                 currenttime = time.time() * 1000 #time in miliseconds.
    #                 print("smliked", currenttime, lasttime)
    #                 if (currenttime - lasttime) > debouncetime:
    #                     print(smiled)
    #                     smiled = smiled + 1
    #                     lasttime = currenttime

    #             previouslysmiling = issmiling(liplms)
#     

            # printing smiled(smliked) whenever detected, error in timing.

            # mpdraw.draw_landmarks(flippedimg, facelms, mpfacemesh.FACEMESH_CONTOURS, drawspec)


    

def main():
    cap = cv2.VideoCapture(0)
    detector = Smile_Detector()

    while True:
        success, img = cap.read()
        flippedimg = cv2.flip(img, 1)

        img = detector.lipemesh(img, True)

        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()


# trying to show video with mesh firstly.