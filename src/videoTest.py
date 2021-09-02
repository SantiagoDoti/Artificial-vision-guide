import os.path
import cv2
import ImageProcessor
import Utils

rootPath = os.path.abspath(os.path.dirname(__file__))
videoPath = os.path.join(rootPath, "../tests/testWhiteRight.mp4")


def go_backward(text):
    text = "delante"
    print("Backward - Marcha atras")


def go_forward(text):
    text = "atras"
    print("Forward - Hacia delante")


def go_left(speed, text):
    text = "izquierda"
    print("Left - Hacia la izquierda")


def go_right(speed, text):
    text = "derecha"
    print("Right - Hacia la derecha")


def stop(text):
    text = "detenido"
    print("Stop - Frenando")


# Video pregrabado
video = cv2.VideoCapture(videoPath)
pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)

initialTrackbarVals = [345, 346, 80, 519]
Utils.initializeTrackbars(initialTrackbarVals)

initialTrackbarVals2 = [1, 50, 40, 60]
Utils.initializeTrackbars2(initialTrackbarVals2)

while True:
    flag, frame = video.read()

    # Loop video
    if not flag:
        video = cv2.VideoCapture(videoPath)
        continue

    Utils.print_base_text(frame)

    edges_cropped_image = ImageProcessor.setup_edges_image(frame)
    lane_detection_image = ImageProcessor.setup_lane_detection_image(frame, Utils.valTrackbars2())
    # warped_image = ImageProcessor.setup_warped_image(frame)
    # warped_points_image = ImageProcessor.setup_warped_points_image(frame)

    cv2.imshow("Lane detection", lane_detection_image)
    cv2.imshow("Edges image", edges_cropped_image)
    # cv2.imshow("Warped image", warped_image)
    # cv2.imshow("Warp points", warped_points_image)

    # pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
    # print(str(pos_frame) + " frames y " + str(video.get(cv2.CAP_PROP_POS_MSEC)/1000) + " segundos")

    if cv2.waitKey(10) == 27:
        break

video.release()
cv2.destroyAllWindows()
