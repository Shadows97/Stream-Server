import cv2
from base_camera import BaseCamera
import threading

# In[]
class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False
		
        self.capture = cv2.VideoCapture('rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov')
        #assert self.capture.isOpened(), "Unable to read the camera"
         
        # initialization
        self.framewidth = int(self.capture.get(3))
        self.frameheight = int(self.capture.get(4))
        print("frame width:{}, height:{}".format(self.framewidth, self.frameheight))
        
        # find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        
        # get fps
        if int(major_ver) < 3 :
            #self.capture.set(cv2.cv.CV_CAP_PROP_FPS, self.setfps)
            fps = self.capture.get(cv2.cv.CV_CAP_PROP_FPS)
        else :
            #self.capture.set(cv2.CAP_PROP_FPS, self.setfps)
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

class Camera(BaseCamera):
    
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        
        import numpy as np
        count = 0
        
        with ipcamCapture(0) as cam: 
            while True:
                state, frame = cam.getframe()
                if state:
                    yield cv2.imencode('.jpg', frame)[1].tobytes()
                else:
                    count = (count + 1) % 255
                    img = np.zeros((240,480,3))
                    img.fill(count)
                    yield cv2.imencode('.jpg', img)[1].tobytes()












