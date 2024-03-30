import cv2
import mediapipe as mp
import pyautogui as pag
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def close_prog(lm):
    f4_x, f4_y, _1, _2 = lm[4]
    f16_x, f16_y, _1, _2 = lm[16]
    if abs(f4_x - f16_x) < 20 and abs(f4_y - f16_y) < 20:
        return True
    else:
        return False

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7, maxHands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0, None)
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
area = 0
colorVol = (255, 0, 0)

while True:
    success, img = cap.read()
    
    (h, w) = img.shape[:2]
    # center = (w / 2, h / 2)
    # M = cv2.getRotationMatrix2D(center, 180, 1.0)
    # img = cv2.warpAffine(img, M, (w, h))

    imgBGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = hands.process(imgBGR)
    
    

    # Find Hand
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    if len(lmList) != 0:

        # Filter based on size
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
        # print(area)
        if 250 < area < 1000:
            # print("yes")

            # Find distance between index and Thumb
            length, img, lineInfo = detector.findDistance(4, 8, img)

            # Convert Volume
            volBar = np.interp(length, [50, 200], [400, 150])
            volPer = np.interp(length, [50, 200], [0, 100])

            # Reduce Resolution to make it smoother
            smoothness = 5
            volPer = smoothness * round(volPer / smoothness)

            # Check fingers up
            fingers = detector.fingersUp()
            
            
            # If pinky is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                colorVol = (255, 255, 0)
                time.sleep(0.05)
            else:
                colorVol = (255, 0, 0)
                

    # Drawings
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Volume set: {int(cVol)}', (600, 50), cv2.FONT_HERSHEY_COMPLEX, 1, colorVol, 3)
    
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            landmark = [(int(point.x * w), int(point.y * h), int(point.x * 1900), int(point.y * 1850)) for point in
                        handLms.landmark]
            
            

            if close_prog(landmark):
                exit()

            x_finger, y_finger, x_mouse, y_mouse = landmark[0]
            pag.moveTo(x_mouse, y_mouse)

            f4_x, f4_y, _1, _2 = landmark[4]
            f12_x, f12_y, _1, _2 = landmark[12]
            if abs(f4_x - f12_x) < 20 and abs(f4_y - f12_y) < 20:
                pag.click()
                print(f'click  {abs(f4_x - f12_x)=},  {abs(f4_y - f12_y)=}')
                


    # Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 100), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
 
