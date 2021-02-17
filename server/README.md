# Flask Video Streaming for Object Detection



The video streaming over flask is referred from https://github.com/miguelgrinberg/flask-video-streaming, the `Montion JPG` idea. More details about it please refer to [video streaming with Flask](http://blog.miguelgrinberg.com/post/video-streaming-with-flask) and follow-up of the previous article [Flask Video Streaming Revisited](http://blog.miguelgrinberg.com/post/flask-video-streaming-revisited).

The second part is object detection on tensorflow. We combine them in order to deploy on the device or the environment without display, for example, IoT devices or container, etc.

The model is `ssd_mobilenet_v1_coco` referred from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md.



## Quick Start



### Developing Mode

* Run a simple streaming example from a video file over flask.

```sh
$ python app.py
```

* Run a streaming example from a web camera over flask. In windows,  run `set CAMERA=opencv` first.

```sh
$ CAMERA=opencv python app.py
```

* Run a realtime object detection example from a web camera over flask. In windows,  run `set CAMERA=obdet` first.

```sh
$ CAMERA=obdet python app.py
```



After start the service, you can surf the web on `http://localhost:5000`.



### Production Mode

Here we use `Gunicorn` tool to deploy our services, notice that windows platform did not support it.

Notice if you run the following scripts using a **external** webcam on macbook, it might go wrong due to `AVCaptureSession warning: Session received the following error while decompressing video: Error Domain=NSOSStatusErrorDomain Code=-12903`.

* [**ubuntu**] Run based on the threading worker. 

```sh
# --threads: allow how many clients access to the flask server
# --worker-class: default is sync
$ CAMERA=opencv gunicorn --threads 5 --workers 1 --bind 0.0.0.0:5000 app:app
```

* [**ubuntu**] Run based on un-sync concurrency. 

```sh
# worker type: eventlet, gevent, tarnado, etc.
$ CAMERA=opencv gunicorn --worker-class gthread --threads 5 --workers 1 --bind 0.0.0.0:5000 app:app
$ CAMERA=obdet gunicorn --worker-class gthread --threads 5 --workers 1 --bind 0.0.0.0:5000 app:app
```

* [**Macbook**] Run the script using **internel camera only**, not for externel camera.

```sh
$ CAMERA=opencv gunicorn --worker-class gthread --threads 5 --workers 1 --bind 0.0.0.0:5000 app:app
```



After start the service, you can surf the web on `http://<IP-address>:5000`.