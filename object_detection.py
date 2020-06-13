import cv2
import numpy as np

# Loading YOLO Algorithm
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
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

# Loading image
image = cv2.imread("gorakhpur_crossroad.jpg")
# image = cv2.imread("Jaipur_pinkcity.jpg")
# image = cv2.imread("sample_img.jpg")
image = cv2.resize(image, None, fx=0.4, fy=0.3)
height, width, channels = image.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# for x in blob :
    # for n, img in enumerate(x) :
        # cv2.imshow(str(n), img)

net.setInput(blob)                 # passing the image into the network and do the detection
outs = net.forward(layers_output)  # an array containing all the informations about objects detected, their position and the confidence about the detection    
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
        if confidence > 0.5 :
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
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
for i in range(objects_detected) :
    if i in indexes :
        x,y,w,h = rectangle_boxes[i]
        label = str(objects[ids[i]])
        # print(label)
        color = colors[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x,y+25), font, 1, color, 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()