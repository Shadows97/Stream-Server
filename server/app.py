#!/usr/bin/env python
from importlib import import_module
import os
from flask import Flask, render_template, Response
from kafka import KafkaConsumer
from json import loads




# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
    print("Load module {}".format('camera_' + os.environ['CAMERA']))
else:
    from camera import Camera

# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    
    app.run(host='0.0.0.0', threaded=True, debug=True)
    consumer = KafkaConsumer('test',auto_offset_reset='earliest',enable_auto_commit=True,value_deserializer=lambda m: loads(m.decode('utf-8')),api_version=(0, 10, 1),bootstrap_servers=['kafka:9092'])
    for m in consumer:
        print(m.value)
    
    
