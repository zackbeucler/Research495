######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
import threading
from threading import *
import importlib.util
from gpiozero import LED
from workstation4 import Workstation

#GPIO pins 
Red_LED = LED(24)
Red_LED.off()
Green_LED = LED(25)
Green_LED.off()
LED_Strip = LED(26)
LED_Strip.on()


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
                # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

MODEL_NAME = 'Sample_TFLite_model'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = 0.5
resW, resH = '1280x720'.split('x')
imW, imH = int(resW), int(resH)
use_TPU = True
# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    
# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

#This is initialized in the workstation class
#status = {"dirty":False, "ready":False, "in_use":False}

# TURN ON GREEN LED
Green_LED.on()

ws1 = Workstation(1)
ws2 = Workstation(2)
ws3 = Workstation(3)

while True:
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    # Grab frame from video stream
    frame1 = videostream.read()
    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    #new
    #Workstation 1
    circlex1 = 200
    circley1 = 400
    radius = 200
    
    cv2.circle(frame, (circlex1,circley1), radius, (255, 0, 0), 5) # draw circle
    
    #Workstation 2
    circlex2 = 1150
    circley2 = 400
    
    cv2.circle(frame, (circlex2,circley2), radius, (0, 255, 0), 5) # draw circle
    
    #Workstation 3
    circlex3 = 700
    circley3 = 400
    
    cv2.circle(frame, (circlex3,circley3), radius, (0, 0, 255), 5) # draw circle
    
    object_name = 'null'
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            ymax = int(ymax/1.25)
            

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            # here it's not updating the 'person' label to null or anyother variable so it doesn't go to the else statement
            
            if object_name == 'person':
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                midx,midy = int((xmax+xmin)/2), int((ymax+ymin)/2) # get mid points of bounding boxes
                cv2.circle(frame, (midx,midy), radius=1, color=(0, 0, 0), thickness=3) # draw midpoint
                midcoords = "%s %s" % (midx, midy)
                #print("midpoint coordinates: ", midcoords)
                
                #Checking for workstation 1
                #Maybe stop cleaning when the person is detected again
                if radius >= int((((midx-circlex1)**2) + ((midy-circley1)**2))**0.5): # detect if midpoint of person is in circle
                    ws1.action('person') #updates multiple status 'ready': false 'in_use':true 'waited':false 
                    #ws1.update('dirty', True) # here maybe add another function that times how long the person uses the wks
                
                if radius >= int((((midx-circlex2)**2) + ((midy-circley2)**2))**0.5):
                    ws2.action('person')
                
                if radius >= int((((midx-circlex3)**2) + ((midy-circley3)**2))**0.5):
                    ws3.action('person')
                    
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            
        if object_name != 'person':
            ws1.action('no_person') #updates status 'dirty': true 'in_use':false 
            ws2.action('no_person')
            ws3.action('no_person')
        
        #checks if the person is not there and updates status 'in_wait': true  then starts the wait thread
        ws1.action('wait')
        ws2.action('wait')
        ws3.action('wait')
            
        #-------------------------------------------------------#
        #double check to see if the person re-enters  <-would there be a better way to code for this part?    
        if ws1.getStatus('waited'):   
            ws1.update('ready',True)
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0) and (labels[int(classes[i])] == 'person')):
                    print("Person has re-entered room----------"+str(ws1.name))
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    ymax = int(ymax/1.25)

                    #what if we don't have new mid?
                    midx, midy = int((xmax+xmin)/2), int((ymax+ymin)/2) # get mid points of bounding boxes
                    cv2.circle(frame, (midx,midy), radius=1, color=(0, 0, 0), thickness=3) # draw midpoint
                            
                    if radius >= int((((midx-circlex1)**2) + ((midy-circley1)**2))**0.5):
                        ws1.action('person')
        
        if ws2.getStatus('waited'):   
            ws2.update('ready',True)
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0) and (labels[int(classes[i])] == 'person')):
                    print("Person has re-entered room----------"+str(ws2.name))
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    ymax = int(ymax/1.25)

                    #what if we don't have new mid?
                    midx, midy = int((xmax+xmin)/2), int((ymax+ymin)/2) # get mid points of bounding boxes
                    cv2.circle(frame, (midx,midy), radius=1, color=(0, 0, 0), thickness=3) # draw midpoint
                        
                    if radius >= int((((midx-circlex2)**2) + ((midy-circley2)**2))**0.5):
                        ws2.action('person')
        
        if ws3.getStatus('waited'):   
            ws3.update('ready',True)
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0) and (labels[int(classes[i])] == 'person')):
                    print("Person has re-entered room----------"+str(ws3.name))
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    ymax = int(ymax/1.25)

                    #what if we don't have new mid?
                    midx, midy = int((xmax+xmin)/2), int((ymax+ymin)/2) # get mid points of bounding boxes
                    cv2.circle(frame, (midx,midy), radius=1, color=(0, 0, 0), thickness=3) # draw midpoint
                        
                    if radius >= int((((midx-circlex3)**2) + ((midy-circley3)**2))**0.5):
                        ws3.action('person')
        #---------------------------------------------------------#
        

        ws1.action('clean')
        ws2.action('clean')
        ws3.action('clean') 
    
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
    
    #print('ws1'+ws1.getAllStatus())
    #print('ws2**'+ws2.getAllStatus())
    #print('ws3**'+ws3.getAllStatus())
# Clean up
cv2.destroyAllWindows()
videostream.stop()


