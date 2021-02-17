
from base_camera import BaseCamera
import cv2

# In[]

class Camera(BaseCamera):
    
    @staticmethod
    def frames():
        
        while True:
        
            cam = cv2.VideoCapture("video.MOV")
            assert cam.isOpened(), "Can not open video file."
            
            while True:
                ret, img = cam.read()
                
                if ret:
                    # resize the frame to 0.5x
                    img = cv2.resize(img, (0, 0), None, .5, .5)
                    
                    # encode as a jpeg image and return it
                    yield cv2.imencode('.jpg', img)[1].tobytes()
                else:
                    break
            
