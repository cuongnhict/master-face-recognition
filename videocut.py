import numpy as np
import cv2
from glob import glob


input_path = './data/video/mica-cam-output.mp4'
video_cap = cv2.VideoCapture(input_path)

while(video_cap.isOpened()):
    ret, frame = video_cap.read()
    if not ret:
        break
    cv2.imshow('img', frame)
    cv2.waitKey()

video_cap.release()
cv2.destroyAllWindows()
