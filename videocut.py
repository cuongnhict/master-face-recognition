import numpy as np
import cv2
from glob import glob


input_path = './data/video/mica-cam-output.mp4'
video_cap = cv2.VideoCapture(input_path)

output_path = './data/video/mica-cam-cut.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, (1920, 1080))
idx = 1

while(video_cap.isOpened()):
    print(idx)
    idx += 1
    ret, frame = video_cap.read()
    if not ret:
        break
    out.write(frame)
    if idx == 1600:
        break

video_cap.release()
cv2.destroyAllWindows()
# out.release()
