import cv2 as cv
import mediapipe as mp
import time
import HandTrackingModel as htm

pTime = 0
cTime = 0
cam = cv.VideoCapture(0)
detector = htm.handDetector()
while True:
    isTrue, frame = cam.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame)
    if len(lmList) != 0:
        print(lmList[4])
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(frame, str(int(fps)), (10, 70),
               cv.FONT_HERSHEY_PLAIN, 3, 255, 2)

    cv.imshow('Cam', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cam.release()
cv.destroyAllWindows()
