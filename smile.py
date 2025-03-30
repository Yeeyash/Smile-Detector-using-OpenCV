import cv2
import mediapipe as mp


# primary camera was "0" if you want any other source, change the index inside "VideoCapture".
cap = cv2.VideoCapture(0)

mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()
    flippedimg = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(flippedimg, cv2.COLOR_BGR2RGB)
    results = hands.process(imgrgb)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = flippedimg.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
            
            
            mpdraw.draw_landmarks(flippedimg, handLms, mphands.HAND_CONNECTIONS)

    cv2.imshow("Image", flippedimg)
    cv2.waitKey(1)