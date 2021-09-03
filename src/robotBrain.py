import cv2
import time
import ImageProcessor
import Utils
# import MotorHandler

# Video en directo de la webcam
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    Utils.print_info_text(frame)

    lane_detection_image = ImageProcessor.setup_lane_detection_image(frame, None)
    # cropped_image = ImageProcessor.setup_edges_image(frame)

    cv2.imshow("Lane detection image", lane_detection_image)
    # cv2.imshow("Edges image (cropped)", cropped_image)

    # MotorHandler.go_left(0.5)
    # time.sleep(3)
    # MotorHandler.stop()

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
