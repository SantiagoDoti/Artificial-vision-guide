import cv2
import numpy as np
from gpiozero import Robot
import RPi.GPIO as GPIO
import time

""" ************************** 
     Escritura sobre pantalla  
    ************************** """


def print_base_text(imagen, speed, direction):
    cv2.rectangle(imagen, (10, 10), (310, 100), (0, 0, 255), 4)
    cv2.putText(imagen, "Rumbo: ", (18, 50), 2, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imagen, direction, (150, 50), 3, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(imagen, "Velocidad: ", (18, 80), 2, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imagen, str(speed), (190, 80), 3, 1, (255, 255, 255), 2, cv2.LINE_AA)


""" ************************* 
     Procesamiento de imagen  
    ************************* """

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
    lines_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_image


def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    mask = cv2.inRange(img, low_white, up_white)
    return cv2.Canny(mask, 75, 150)


""" ******************* 
     Manejo de motores  
    ******************* """

# L298N pines <> GPIO pines
IN1, IN2, IN3, IN4, EN = 27, 22, 23, 24, 25

# def setup_motors():
#     GPIO.setmode(GPIO.BCM)
#     GPIO.setup(EN, GPIO.OUT)
#     GPIO.setup(IN1, GPIO.OUT)
#     GPIO.setup(IN2, GPIO.OUT)
#     GPIO.setup(IN3, GPIO.OUT)
#     GPIO.setup(IN4, GPIO.OUT)
#     GPIO.output(IN1, GPIO.LOW)
#     GPIO.output(IN2, GPIO.LOW)
#     GPIO.output(IN3, GPIO.LOW)
#     GPIO.output(IN4, GPIO.LOW)
#     print("Setup motors - Preparando los motores")


def go_backward(text):
    mr_robot.backward()
    text = "delante"
    print("Backward - Marcha atras")


def go_forward(text):
    mr_robot.forward()
    text = "atras"
    print("Forward - Hacia delante")


def go_left(speed, text):
    mr_robot.left(0.5)
    text = "izquierda"
    print("Left - Hacia la izquierda")


def go_right(speed, text):
    mr_robot.right(0.5)
    text = "derecha"
    print("Right - Hacia la derecha")


def stop(text):
    mr_robot.stop()
    text = "detenido"
    print("Stop - Frenando")


""" *********************
     Ejecución principal  
    ********************* """

# Video en directo de la webcam
video = cv2.VideoCapture(0)

mr_robot = Robot(left=(IN1, IN2), right=(IN3, IN4))
robot_speed_text = 0.5
robot_direction_text = "detenido"

while True:
    check, frame = video.read()

    print_base_text(frame, robot_speed_text, robot_direction_text)

    processedImage = process_image(frame)

    croppedImage = region_of_interest(processedImage)

    # lines = cv2.HoughLinesP(img, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
    lines = cv2.HoughLinesP(croppedImage, rho=2, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=20, maxLineGap=40)
    linesImage = draw_lines(frame, lines)

    finalImage = cv2.addWeighted(frame, 0.8, linesImage, 1, 1)

    cv2.imshow("Resultado", finalImage)
    # cv2.imshow("Edges image (cropped)", croppedImage)

    go_left(robot_speed_text, robot_direction_text)
    time.sleep(3)
    stop(robot_direction_text)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
