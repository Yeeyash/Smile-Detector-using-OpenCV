import cv2
import mediapipe as mp
import time

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
        
        self.currentlysimling = False
        self.previouslysmiling = False

        self.lasttime = 0
        self.debouncetime = 1000
        self.smiled = 0

        self.upperlistindices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.lowerlipindices = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        self.innerupperlistindices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        self.innerlowerlipindices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 317, 14]

    def lipemesh(self, img, draw= True):
        self.imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.facemesh.process(self.imgrgb)
        ih, iw, _ = img.shape
        liplms = {}

        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks: #lms - landmarks

                alllipindx = self.upperlistindices + self.lowerlipindices + self.innerlowerlipindices + self.innerupperlistindices
                for idx in alllipindx:
                    landmark = facelms.landmark[idx]
                    x, y = int(landmark.x * iw), int(landmark.y * ih)

                    liplms[idx] = (x,y)
                    cv2.circle(img, (x,y), 2, (0,255,0), -1)
            
                return img, facelms


        return img, None
    
    def issmiling(self, facelms):
        left_mouth_corner = facelms.landmark[61]
        right_mouth_corner = facelms.landmark[291]
        upper_lip_top = facelms.landmark[0]
        lower_lip_bottom = facelms.landmark[17]

        mouth_width = (abs(right_mouth_corner.x - left_mouth_corner.x))
        mouth_height = (abs(upper_lip_top.y - lower_lip_bottom.y))
                
        mouth_ratio = mouth_width/mouth_height if mouth_height != 0 else 0

        return mouth_ratio > 2.5 and mouth_height*100 > 3.5


    def smiledcounter(self, facelms):
        if self.issmiling(facelms):
            self.smiled = self.smiled + 1
            # return self.smiled
            print(self.smiled)


    def main(self):
        cap = cv2.VideoCapture(0)
        detector = Smile_Detector()

        while True:
            success, img = cap.read()
            flippedimg = cv2.flip(img, 1)
            current_smiles = 0
            flippedimg, facelms = self.lipemesh(flippedimg, True)

            if facelms:
                currentlysmiling = self.issmiling(facelms)
                
                if currentlysmiling != self.previouslysmiling:
                    currenttime = time.time() * 1000
                    if (currenttime - self.lasttime) > self.debouncetime:
                        self.smiledcounter(facelms)
                        # self.smiled = self.smiled + 1
                        # print(self.smiled, currenttime, self.lasttime)
                        # return self.smiled
                        self.lasttime = currenttime
                current_smiles = self.smiled
                self.previouslysmiling = currentlysmiling

            _, buffer = cv2.imencode('.jpg', flippedimg)
            frame = buffer.tobytes()

            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # cv2.imshow("image", flippedimg)
            # cv2.waitKey(1)


if __name__ == "__main__":
    detector = Smile_Detector()
    detector.main()



# trying to show video with mesh firstly.