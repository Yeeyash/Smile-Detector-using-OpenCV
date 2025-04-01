import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

# import mediapipe classes/functions for face detection.

mpfacedetect = mp.solutions.face_detection
# drawing_utils is used to detect landmarks and connections by its model. 
mpdraw = mp.solutions.drawing_utils
# .FaceDetection allows us to use the face detection model.
facedetection = mpfacedetect.FaceDetection()

while True:
    success, img = cap.read()
    flippedimg = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(flippedimg, cv2.COLOR_BGR2RGB)
    results = facedetection.process(imgrgb)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # id is the id number, detection is the values that is recieved from video, this includes location data(data about bounding box, height, width, xmin and ymin) and the keypoints of a face.
            # print(id, detection)
            # to only get the information about bounding box, call the property ".relative_bounding_box" on detection.location_data.
            # print(detection.location_data.relative_bounding_box)

            # since the call is too long, we assign a variable.

            bbox = detection.location_data.relative_bounding_box

            # to draw landmarks over the face, use draw_detection property over mpdraw class.
            # mpdraw.draw_detection(flippedimg, detection)

            # to draw a box using our values:
            ih, iw, ic = img.shape
            bboximg = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                        int(bbox.width * iw), int(bbox.height * ih)

            cv2.rectangle(flippedimg, bboximg, (255, 0 ,255), 2)
            # there are a few false detections, this can be reduced by changing the value of "minimum confidence value" in facedetection class.
            # to show the confidence score, use ".puttext()" feature.

            cv2.putText(flippedimg, f"{int(detection.score[0]*100)}%", (bboximg[0], bboximg[1] - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", flippedimg)
    cv2.waitKey(1)

