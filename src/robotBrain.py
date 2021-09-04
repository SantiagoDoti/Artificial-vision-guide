import cv2
import time
import ImageProcessor
import Utils
import MotorHandler
from picamera.array import PiRGBArray
from picamera import PiCamera

# Video en directo desde la Raspberry Pi Camera
camera = PiCamera()
camera.resolution = (720, 480)
#camera.framerate = 32
raw_capture = PiRGBArray(camera, size=(720, 480))
       
time.sleep(0.1)

for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    image = frame.array

    #cropped_image = ImageProcessor.setup_edges_image(image)
    lane_detection_image = ImageProcessor.setup_lane_detection_image(image, None)

    cv2.imshow("Lane detection image", lane_detection_image)
    #cv2.imshow("Edges image (cropped)", cropped_image)

    #MotorHandler.go_right(0.5,image)
    
    #cv2.imshow("Frames", image)


    raw_capture.truncate(0)
    
    if cv2.waitKey(1) == 27:
        MotorHandler.shutdown_motor()
        break

# video.release()
cv2.destroyAllWindows()
