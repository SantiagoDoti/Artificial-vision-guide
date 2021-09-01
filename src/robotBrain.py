import cv2
import time
import ImageProcessor
import Utils
# import MotorHandler

# Video en directo de la webcam
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    Utils.print_base_text(frame)

    final_image = ImageProcessor.setup_image(frame, None)

    cv2.imshow("Resultado", final_image)
    # cv2.imshow("Edges image (cropped)", croppedImage)

    # MotorHandler.go_left(0.5)
    # time.sleep(3)
    # MotorHandler.stop()

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
