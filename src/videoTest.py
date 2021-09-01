import os.path
import cv2
import numpy as np

rootPath = os.path.abspath(os.path.dirname(__file__))
videoPath = os.path.join(rootPath, "../tests/testWhiteRight.mp4")


def print_base_text(imagen, speed, direction):
    cv2.rectangle(imagen, (10, 10), (310, 100), (0, 0, 255), 4)
    cv2.putText(imagen, "Rumbo: ", (18, 50), 2, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imagen, direction, (150, 50), 3, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(imagen, "Velocidad: ", (18, 80), 2, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imagen, str(speed), (190, 80), 3, 1, (255, 255, 255), 2, cv2.LINE_AA)


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


# Filter colors
low_white = np.array([5, 5, 160])
up_white = np.array([179, 85, 235])
low_yellow = np.array([18, 94, 140])
up_yellow = np.array([48, 255, 255])


def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    triangle = [np.array([(150, height), (850, height), (450, 320)])]
    trapeze = [np.array([(0, height), (width / 2, height / 2), (width, height)], dtype=np.int32)]
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, trapeze, 255)
    return cv2.bitwise_and(img, mask)


def draw_lines(img, lines):
    # Creamos otra imagen totalmente negra del mismo tamaño que la original para dibujar las lineas sobre ella
    lines_image_black = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_image_black


def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gaussian_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    mask = cv2.inRange(gaussian_image, low_white, up_white)
    return cv2.Canny(mask, 75, 150)


# Video pregrabado
video = cv2.VideoCapture(videoPath)
robot_speed_text = 0.5
robot_direction_text = "detenido"

pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)

while True:
    flag, frame = video.read()

    # Loop video
    if not flag:
        video = cv2.VideoCapture(videoPath)
        continue

    print_base_text(frame, robot_speed_text, robot_direction_text)

    # Aplicamos varios filtros a la imagen y luego detectamos los bordes
    processedImage = process_image(frame)

    # Recortamos la imagen (ROI) para reducir el ruido y los falsos positivos
    cropped_image = region_of_interest(processedImage)

    # Creamos las lineas sobre la imagen negra cortada
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=20, maxLineGap=40)
    lines_image = draw_lines(frame, lines)

    # Combinamos la imagen negra con las lineas dibujadas contra la del video para obtener la imagen final
    final_image = cv2.addWeighted(frame, 0.8, lines_image, 1, 1)

    # cv2.imshow("Video original", frame)
    cv2.imshow("Resultado", final_image)
    # cv2.imshow("Edges image (cropped)", cropped_image)
    pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
    # print(str(pos_frame) + " frames y " + str(video.get(cv2.CAP_PROP_POS_MSEC)/1000) + " segundos")

    if cv2.waitKey(10) == 27:
        break

video.release()
cv2.destroyAllWindows()
