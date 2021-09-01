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

while True:
    flag, frame = video.read()

    # Loop video
    if not flag:
        video = cv2.VideoCapture(videoPath)
        continue

    Utils.print_base_text(frame)

    final_image = ImageProcessor.setup_image(frame)

    # cv2.imshow("Video original", frame)
    cv2.imshow("Resultado", final_image)
    # cv2.imshow("Edges image (cropped)", cropped_image)
    pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
    # print(str(pos_frame) + " frames y " + str(video.get(cv2.CAP_PROP_POS_MSEC)/1000) + " segundos")

    if cv2.waitKey(10) == 27:
        break

video.release()
cv2.destroyAllWindows()
