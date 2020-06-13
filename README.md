# Real-Time-Object-Detection-using-CNNs-YOLO
This repository provides an insight into the project done on real time object detection in CPU using a deep learning object detection algorithm i.e., YOLO.

YOLO is a clever convolutional neural network (CNN) for doing object detection in real-time. The algorithm applies a single neural network to the full image, and then divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.
YOLO (You Only Look Once) is a deep learning object detection algorithm which is used in this project with the help of OpenCV library.


This project contains :-

* Two configuration files containing the details of the model inside a cfg folder.
-> yolov3.cfg
-> yolov3-tiny.cfg

* A names file having the dataset containing the names of objects.
-> obj.names

* Two python files containing the source code.
-> object_detection.py ------ For detection of objects in an image.
-> realtime_detection.py ---- For Real-Time detection in CPU.

* Two weights file containing the pre-trained weights.
-> yolov3.weights ----------- https://pjreddie.com/darknet/yolo/
-> yolov3-tiny.weights ------ https://pjreddie.com/darknet/yolo/


-----------------------------object_detection.py --------------------------
Results in progress :-

----------->
Blob is used to extract feature from the image and resize them.
 1.Blob in red 2.Blob in green 3.Blob in blue
Now, image is ready to be processed in Yolo algorithm . . .

----------->
Image blob is passed into the network to do the detection and the information of the objects detected is passed into an array.

----------->
Objects are visualized on the screen.


-----------------------------realtime_detection.py----------------------------
Results in progress :-

----------->
Loading the camera or the video.

----------->
Getting the starting time and the frame ID in order to calculate FPS of the processing video.

----------->
Extracting the frame from the camera and performing the detection.

----------->
FPS is calculated and everything is visualized and displayed on the screen.

CPU vs GPU
Advantages in CPU - it’s really easy to set up and it works right away on Opencv without doing any further installations. We only need Opencv 3.4.2 or greater.
Challenges in CPU - Very Slow in detecting frames in the video
To overcome speed . . .
-> Reducing the size of blob to (320,320) - This will increase some speed but somehow decrease the accuracy.
-> yolov3-tiny.cfg and yolov3_tiny.weights files - version of yolo optimized for CPUs for increasing speed.








