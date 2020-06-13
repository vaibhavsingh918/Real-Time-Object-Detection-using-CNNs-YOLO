import cv2
import numpy as np
import time

# Loading YOLO Algorithm
net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
objects = []
with open("obj.names", "r") as f:
    for line in f.readlines() :
        objects.append(line.strip())
print(objects)
layers_output=[]
layer_names = net.getLayerNames()
for i in net.getUnconnectedOutLayers() :
    layers_output.append(layer_names[i[0]-1])
colors = np.random.uniform(0, 255, size=(len(objects), 3))

# Loading video
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("sample_vid.mp4")
# cap = cv2.VideoCapture("sample_vid2.mp4")

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
time_start = time.time()
frame_count = 0                  # to count no of frames
while True :
    _, frame = cap.read()
    frame_count = frame_count+1

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    # for x in blob :
        # for n, img in enumerate(x) :
            # cv2.imshow(str(n), img)

    net.setInput(blob)
    outs = net.forward(layers_output)
    # print(outs)

    # Visualizing objects on screen
    rectangle_boxes = []
    confidences = []
    ids =[]
    for i in outs:
        for j in i:
            scores = j[5:]
            iD = np.argmax(scores)
            confidence = scores[iD]
            if confidence > 0.2 :
                x_centre = int(j[0]*width)
                y_centre = int(j[1]*height)
                w = int(j[2]*width)
                h = int(j[3]*height)

                # cv2.circle(image, (x_centre,y_centre), 10, (0, 255, 0), 2)
                # Rectangle coordinates
                x = int(x_centre - w/2)
                y = int(y_centre - h/2)

                # cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
                rectangle_boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                ids.append(iD)

    indexes = cv2.dnn.NMSBoxes(rectangle_boxes, confidences, 0.5, 0.4)     #to store indexes of objects uniquely
    # print(indexes)
    objects_detected = len(rectangle_boxes)

    for i in range(objects_detected) :
        if i in indexes :
            x,y,w,h = rectangle_boxes[i]
            label = str(objects[ids[i]])
            confidence = confidences[i]
            # print(label)
            color = colors[ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label+" "+str(round(confidence, 2)), (x,y+25), font, 1, color, 2)

    time_elapsed = time.time() - time_start            # time passed
    fps = frame_count / time_elapsed                   # no. of frames processing per second
    cv2.putText(frame, "FPS: "+str(round(fps, 2)), (10, 30), font, 2, (255, 255, 255), 1)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27 :
        break

cv2.release()
cv2.destroyAllWindows()