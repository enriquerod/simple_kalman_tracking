# from kalman_filter import kalmanTrack
from Kalman_filter import KalmanTrack
from Comp_vision import Comp_vision
from numpy import * 
import cv2

#import numpy as np

kalman = KalmanTrack()
obj_pos = Comp_vision()

background = cv2.imread('frames/bg.jpg')

for i in range(23):
    path = 'frames/' + str(i+1) + '.jpg'
    frame = cv2.imread(path)

    x , y = obj_pos.diff_backg(frame, background)
    print(x, y)
    xh, yh = kalman(x, y)
    print(xh, yh)
    xh = int(xh)
    yh = int(yh)
    x = int(x)
    y = int(y)
    
    cv2.rectangle(frame, (xh -13, yh - 13), (xh + 13, yh + 13), (0, 0, 255), 2)
    cv2.circle(frame,(xh,yh), 2, (0,0,255), -1)

    cv2.rectangle(frame, (x - 13, y - 13), (x + 13, y + 13), (255, 0, 0), 2)
    cv2.circle(frame,(x,y), 2, (255,0,0), -1)


    cv2.imshow("Result Kalman", frame)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()


