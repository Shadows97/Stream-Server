#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
"""

# In[]

import numpy as np
import cv2
import os
import tensorflow as tf
import threading
import time
from collections import OrderedDict
import re
import random
from base_camera import BaseCamera
from importlib import import_module
from kafka import KafkaProducer
from json import dumps
import stomp
# from pykafka import kafkaClient

producer = KafkaProducer(
   value_serializer=lambda m: dumps(m).encode('utf-8'), 
   bootstrap_servers='kafka:9092')

# client = kafkaClient(host='kafka:9092')
# topic = client.topics['test']
# producers = topic.get_sync_producer()

user = "admin"
password =  "password"
host = "activemq"
port =  61616
destination = ["/topic/event"]
destination = destination[0]

conn = stomp.Connection(host_and_ports = [(host, port)])
conn.start()
conn.connect(login=user,passcode=password)



print("Tensorflow version: {}".format(tf.__version__))

# In[]

# =============================================================
# for a begineer, this is the only parameter you need to change
# =============================================================
your_own_model = False

# In[]

"""
PB: the frozen model
pb.txt: the label map file
"""

if your_own_model:
    RETRAINED_MODEL_PATH = "/notebooks/object_detection/model"
    PATH_TO_PB = os.path.join(RETRAINED_MODEL_PATH, "frozen_inference_graph.pb")
    PATH_TO_LABELS = os.path.join(RETRAINED_MODEL_PATH, "label_map.pbtxt")
else:
    PRETRAINED_MODEL_PATH = "./model/ssd_mobilenet_v1_coco_2018_01_28/"
    PATH_TO_PB = os.path.join(PRETRAINED_MODEL_PATH, "frozen_inference_graph.pb")
    PATH_TO_LABELS = os.path.join(PRETRAINED_MODEL_PATH, "mscoco_label_map.pbtxt")

if not os.path.exists(PATH_TO_PB): raise FileNotFoundError("PB is not found.")
if not os.path.exists(PATH_TO_LABELS): raise FileNotFoundError("Label is not found")

# In[]

# load graph
graph = tf.Graph()
with graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_PB, 'rb') as fid:
        serialized_graph = fid.read()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')
        
# In[]
        
category_index = OrderedDict()
with open(PATH_TO_LABELS, "r") as fin:
    tmpData = ""
    for line in fin:
        tmpData += line.strip()
    #print(tmpData)
    
    if your_own_model:
        pattern = re.compile("item\s+\{id:\s+(\d*)name:\s+'([\d\S]*)'\}", re.I)
        allItems = pattern.findall(tmpData)
        for item_idx in range(len(allItems)):
            category_index[int(allItems[item_idx][0])] = {'id': int(allItems[item_idx][0]), \
                                                          'name': str(allItems[item_idx][1])}
    else:
        pattern = re.compile("item\s+\{name:\s+\"([\d\S]*)\"id:\s+(\d*)display_name:\s+\"([a-zA-Z0-9_ ]*)\"\}", re.I)
        allItems = pattern.findall(tmpData)
        for item_idx in range(len(allItems)):
            category_index[int(allItems[item_idx][1])] = {'id': int(allItems[item_idx][1]), \
                                                          'name': str(allItems[item_idx][2])}
    #print(category_index)

# In[]

# fetch realtime frame from camera 
class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False
		
        #self.capture = cv2.VideoCapture('rtsp://192.168.43.106:8080/h264_ulaw.sdp')
        self.capture = cv2.VideoCapture('rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov')
        assert self.capture.isOpened(), "Unable to read the camera"
         
        # initialization
        self.framewidth = int(self.capture.get(3))
        self.frameheight = int(self.capture.get(4))
        print("frame width:{}, height:{}".format(self.framewidth, self.frameheight))
        
        # find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        
        # get fps
        if int(major_ver) < 3 :
            fps = self.capture.get(cv2.cv.CV_CAP_PROP_FPS)
        else :
            fps = self.capture.get(cv2.CAP_PROP_FPS)
        print("FPS:{}".format(fps))
        
        # auto start
        self.start()
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.stop()
        except Exception as e:
            print("{}".format(str(e)))

    def start(self):
        # daemon=True: stopping while process is stopping
        print('Start capturing realtime streaming!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
        self.isstop = True
        print('Stop capturing!')
   
    def getframe(self):
        # only return the latest frame
        return self.status, self.Frame
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
        self.capture.release()
        
# In[]
        
def color_generator(number_of_colors=90):
    random.seed(3)
    r = [np.random.randint(255) for _ in range(number_of_colors)]
    g = [np.random.randint(255) for _ in range(number_of_colors)]
    b = [np.random.randint(255) for _ in range(number_of_colors)]
    random.shuffle(g)
    random.shuffle(b)
    color = np.stack((r,g,b), axis=-1)
    return color.astype('uint8')
    
# In[]

# (b,g,r)
colorMap = color_generator()
detectionZone = (255, 255, 255)

# select highly confident
threshold = 5e-1

# font style
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
lineType = 2

# In[]

class Camera(BaseCamera):
    
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        with ipcamCapture(Camera.video_source) as cam:    
            
            scaledSize = .8
            cam_h, cam_w = cam.frameheight, cam.framewidth
            cam_h_s, cam_h_e = int(cam_h * 0.1 * scaledSize), int(cam_h * 0.9 * scaledSize)
            cam_w_s, cam_w_e = int(cam_w * 0.1 * scaledSize), int(cam_w * 0.9 * scaledSize)
            # partial capture (y1, x1, y2, x2)
            partialCap = [[cam_h_s, cam_w_s, cam_h_e, cam_w_e]]

            try:
                with graph.as_default():
                    with tf.compat.v1.Session() as sess:
                        
                        # setup input/output layers
                        input_operation = graph.get_tensor_by_name('image_tensor:0')
                        output_operation = {}
                        for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                            tensor_name = key + ':0'
                            output_operation[key] = graph.get_tensor_by_name(tensor_name)
                            
                        start_time = time.time()
                        frame_count = 0
                        frame_rate = 0.0
                        
                        while(True):
                            
                            frame_count += 1
                            time_peroid = time.time() - start_time
                            if time_peroid > 5:
                                frame_rate = (frame_count / int(time_peroid))
                                start_time = time.time()
                                frame_count = 1
                            
                            # capture latest frame
                            state, frame = cam.getframe()
                            
                            if not state: continue
                        
                            # image resize
                            # I just resized the image to a quarter of its original size
                            frame = cv2.resize(frame, (0, 0), None, scaledSize, scaledSize)
                            
                            # plot each boxes
                            for imgIndex in range(len(partialCap)):
                                # region==[height, width, channel]
                                region = frame[partialCap[imgIndex][0]-1:partialCap[imgIndex][2]+1, \
                                        partialCap[imgIndex][1]-1:partialCap[imgIndex][3]+1, :]
                                region_height, region_width, _ = region.shape
                                
                                output_dict = sess.run(output_operation, \
                                        feed_dict={input_operation: np.expand_dims(region, 0)})
                            
                                # convert data type float32 to appropriate
                                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                                
                                odList = np.argwhere(output_dict['detection_scores'] > threshold)
                                odList = odList.reshape(-1)     
                                
                                # show detection region, rectangle((x1, y1), (x2, y2))
                                cv2.rectangle(frame, \
                                      (partialCap[imgIndex][1], partialCap[imgIndex][0]), \
                                      (partialCap[imgIndex][3], partialCap[imgIndex][2]), \
                                      detectionZone, 2)
                                
                                for od in odList:
                                    clsName = category_index[int(output_dict['detection_classes'][od])]['name']
                                    clsPrb = str(round(float(output_dict['detection_scores'][od]) * 100)) + "%"
                                    points = output_dict['detection_boxes'][od]
                                    coloridx = int(output_dict['detection_classes'][od]) % len(colorMap)
                                    colorData = colorMap[coloridx]
                                    colorData = (int(colorData[0]), int(colorData[1]), int(colorData[2]))
                                    y1 = int(round(partialCap[imgIndex][0] + region_height * points[0]))
                                    x1 = int(round(partialCap[imgIndex][1] + region_width * points[1]))
                                    y2 = int(round(partialCap[imgIndex][0] + region_height * points[2]))
                                    x2 = int(round(partialCap[imgIndex][1] + region_width * points[3]))
                                    if str(clsName) == 'person':
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), colorData, 2)
                                        cv2.putText(frame, "{}:{}".format(str(clsName),clsPrb), (x1, y1), 
                                                    font, fontScale, colorData, lineType)
                                        print(clsName)  
                                        conn.send(clsName, destination=destination, persistent='false')     
                                        producer.send('test', {'foo': clsName})
                                        
                                    
                                # print frame rate
                                cv2.putText(frame, "{}: {}".format(\
                                            "FPS", str(round(frame_rate))), 
                                            (int(cam_w_s), int(cam_h_s * 0.9)),
                                            font, 0.6, (255,255,255), 2)
                                
                                    
                            #cv2.imshow('frame',frame)                
                            yield cv2.imencode('.jpg', frame)[1].tobytes()
                            
            except Exception as e:
                print("Inference error:{}".format(str(e)))
                time.sleep(10)
                cam.stop()