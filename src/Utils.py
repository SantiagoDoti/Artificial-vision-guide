import cv2

robot_direction_text = "detenido"
robot_speed = 0


def print_base_text(imagen):
    cv2.rectangle(imagen, (10, 10), (310, 100), (0, 0, 255), 4)
    cv2.putText(imagen, "Rumbo: ", (18, 50), 2, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imagen, robot_direction_text, (150, 50), 3, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(imagen, "Velocidad: ", (18, 80), 2, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imagen, str(robot_speed), (190, 80), 3, 1, (255, 255, 255), 2, cv2.LINE_AA)


def set_robot_direction_text(text):
    text = robot_direction_text


def set_robot_speed(speed):
    speed = robot_speed

