import cv2

import face_detection


def show_img(img, detections):
    for i, detect in enumerate(detections):
        x1, y1 = int(detect[0]), int(detect[1])
        x2, y2 = int(detect[2]), int(detect[3])
        # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(f'./data/tmp/{i+1}.png', img[y1:y2, x1:x2])
        cv2.putText(img, str(i+1), (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    img = cv2.resize(img, (int(img.shape[1]/1.5), int(img.shape[0]/1.5)))
    cv2.imshow('img', img)
    cv2.waitKey()


detector = face_detection.build_detector('RetinaNetMobileNetV1',
                                         confidence_threshold=.5,
                                         nms_iou_threshold=.3)

file_path = './data/video/mica-cam.mp4'
video_cap = cv2.VideoCapture(file_path)
i = 1
while(video_cap.isOpened()):
    ret, frame = video_cap.read()
    if not ret:
        break
    if i % 2 == 0:
        print(i)
        detections = detector.detect(frame)
        show_img(frame, detections)

    i += 1

video_cap.release()
cv2.destroyAllWindows()
