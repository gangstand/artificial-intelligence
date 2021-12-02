import cv2
import numpy as np
import glob
import math

net = cv2.dnn.readNet('artificial_intelligence_files/yolov3.weights', 'artificial_intelligence_files/yolov3.cfg')
classes = []
with open('artificial_intelligence_files/coco.names', 'r') as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture('video/video.mp4')

a = 1000000

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

length_frime_time = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / 15
print('Среднее время выполнения: ' + str(round(length_frime_time)) + ' минут.')


while True:
    length -= 1
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

    cv2.imwrite('video/video_cadr/' + str(a) + '.jpg', img)
    a += 1
    if length < 1:
        break
img_array = []
for filename in glob.glob('video/video_cadr/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 27, size)

for i in range(len(img_array)):
    out.write(img_array[i])

cap.release()
cv2.destroyAllWindows()
