import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

model = YOLO('models/best.pt')

cap = cv2.VideoCapture('datasets/testing/m220.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
tracker = Tracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (405, 720))
    frame = cv2.flip(frame, 1)
    results = model.predict(frame)
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        classes = int(row[5])
        list.append([x1, y1, x2, y2, classes])

    bbox_index = tracker.update(list)
    for box in bbox_index:
        x4, y4, x5, y5, classes, id = box

        xm = (x4 + x5) // 2
        ym = (y4 + y5) // 2

        color = (0, 255, 0) if classes == 0 else (0, 0, 255)
        cv2.circle(frame, (xm, ym), 4, color, -1)

        cv2.line(frame, (xm - 2, ym - 2), (xm + 2, ym + 2), (255, 255, 255), 1)
        cv2.line(frame, (xm - 2, ym + 2), (xm + 2, ym - 2), (255, 255, 255), 1)

    cv2.imshow("FRAME", frame)
    cv2.moveWindow("FRAME", 40, 30)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
