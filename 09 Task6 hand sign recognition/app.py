from handdetector import HandDetector
import cv2


handDetector=HandDetector(min_detection_confidence=0.7)
cam=cv2.VideoCapture(0)
oldDistance=0
while True:
    status,image=cam.read()
    if status:
        result=handDetector.findHandLandMarks(image=image,draw=True)
        print(result)

    cv2.imshow("result",image)
    cv2.waitKey(1)