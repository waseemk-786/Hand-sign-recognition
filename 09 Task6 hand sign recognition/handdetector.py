from cv2 import normalize
import mediapipe as mp
import cv2

#detecting hands
mpHands=mp.solutions.hands
mpDraw=mp.solutions.drawing_utils

class HandDetector:
    def __init__(self,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.hands=mpHands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def storeData(self,f,a):
        f1=open(f,'a')
        f1.write(str(a))
        f1.close()


    def findHandLandMarks(self,image,handNumber=0,draw=False):
        originalImage=image
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        results=self.hands.process(image)
        landMarkList=[]
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for point in mpHands.HandLandmark:
                    normalizeLandmark=hand_landmarks.landmark[point]
                    landMarkList.append(normalizeLandmark.x)
                    landMarkList.append(normalizeLandmark.y)
                    landMarkList.append(normalizeLandmark.z)
                print(len(landMarkList))
                data=str(landMarkList)[1:-1]
                self.storeData('gesture2.csv',data+', gesture2\n')    
            if draw:
                mpDraw.draw_landmarks(originalImage,hand_landmarks,mpHands.HAND_CONNECTIONS)
            return landMarkList