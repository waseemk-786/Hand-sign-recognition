import mediapipe as mp
import pickle
import cv2
import numpy as np


model=pickle.load(open('model.pkl','rb'))

mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
mp_hands=mp.solutions.hands

cam=cv2.VideoCapture(0)
while True:
    status,image=cam.read()
    if status:
        imageWidth,imageHeight=image.shape[:2]
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
            results=hands.process(image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
        
                    )
                    data=[]
                    for point in mp_hands.HandLandmark:
                        normalizeLandmark=hand_landmarks.landmark[point]
                        data.append(normalizeLandmark.x)
                        data.append(normalizeLandmark.y)
                        data.append(normalizeLandmark.z)
                    output=model.predict([data])
                    font=cv2.FONT_HERSHEY_COMPLEX
                    org=(50,50)
                    fontScale=1
                    color=(255,0,0)#blue
                    thickness=2
                    image=cv2.putText(image,output[0],org,font,fontScale,color,thickness,lineType=cv2.LINE_AA)
                    

                    cv2.imshow("result",image)
                    cv2.waitKey(1)