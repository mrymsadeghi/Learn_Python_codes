import os
import cv2 as cv
import numpy as np

yolo_path = "/data/YOLO"

classes = []
with open("/data/YOLO/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

yolo_net = cv.dnn.readNet(os.path.join(yolo_path,"yolov3.weights"), os.path.join(yolo_path,"yolov3.cfg"))
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
colors[0],colors[1],colors[2],colors[5],colors[9] = (176, 104, 176), (226, 43, 138), (255, 191, 0), (153, 255, 255), (0, 136, 255)
cap = cv.VideoCapture("/Videos/driving_camera.mp4")
video_cod = cv.VideoWriter_fourcc(*'XVID')
video_output = cv.VideoWriter('driving_camera_det_yolov3_cv.avi',
                      video_cod,
                      10,
                      (1280,720))
while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        #img = cv.resize(frame, None, fx=0.8, fy=0.8)
        height, width, channels = img.shape
        #print(height, width)
        blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo_net.setInput(blob)
        outs = yolo_net.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv.putText(img, label, (x, y + 30), font, 1, color, 1)
        cv.imshow('Frame',img)
        video_output.write(img)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
video_output.release()
cv.destroyAllWindows()