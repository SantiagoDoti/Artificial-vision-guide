import cv2
import numpy as np

robot_direction = 3     # recto (straigth)
robot_direction_text = "detenido"
robot_speed = 0
wT, hT = 960, 540
# wT, hT = 1280, 720


# Dibujamos sobre la imagen información importante sobre el rumbo actual del robot y la velocidad
def print_base_text(imagen):
    cv2.rectangle(imagen, (10, 10), (310, 100), (0, 0, 255), 4)
    cv2.putText(imagen, "Rumbo: ", (18, 50), 2, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imagen, robot_direction_text, (150, 50), 3, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(imagen, "Velocidad: ", (18, 80), 2, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imagen, str(robot_speed), (190, 80), 3, 1, (255, 255, 255), 2, cv2.LINE_AA)


def print_direction_arrow(image):
    width, height = image.shape[0], image.shape[1]
    if robot_direction == 0:
        pass
    if robot_direction == 1:  # izquierda (left)
        pt1 = (width / 2 - 10, height / 4)
        pt2 = (width / 2 - 30, height / 4)
    elif robot_direction == 2:    # derecha (right)
        pt1 = (width / 2 + 10, height / 4)
        pt2 = (width / 2 + 30, height / 4)
    else:   # recto (straigth)
        pt1 = (width / 2, height / 4)
        pt2 = (width / 2, height / 4 + 30)
    cv2.arrowedLine(image, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255))


def set_robot_direction(direction, text_direction):
    direction = robot_direction
    text_direction = robot_direction_text


def set_robot_speed(speed):
    speed = robot_speed


def nothing(a):
    pass


def initializeTrackbars(initial_tracbar_vals):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)

    cv2.createTrackbar("Width Top", "Trackbars", initial_tracbar_vals[0], wT // 2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", initial_tracbar_vals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", initial_tracbar_vals[2], wT // 2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", initial_tracbar_vals[3], hT, nothing)


def valTrackbars():
    width_top = cv2.getTrackbarPos("Width Top", "Trackbars")
    height_top = cv2.getTrackbarPos("Height Top", "Trackbars")
    width_bottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    height_bottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(width_top, height_top), (wT-width_top, height_top),
                         (width_bottom, height_bottom), (wT-width_bottom, height_bottom)])
    return points


def initializeTrackbars2(initial_tracbar_vals):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)

    cv2.createTrackbar("Rho", "Trackbars", initial_tracbar_vals[0], wT // 2, nothing)
    cv2.createTrackbar("Threshold", "Trackbars", initial_tracbar_vals[1], hT, nothing)
    cv2.createTrackbar("minLineLength", "Trackbars", initial_tracbar_vals[2], wT // 2, nothing)
    cv2.createTrackbar("maxLineGap", "Trackbars", initial_tracbar_vals[3], hT, nothing)


def valTrackbars2():
    rho = cv2.getTrackbarPos("Rho", "Trackbars")
    threshold = cv2.getTrackbarPos("Threshold", "Trackbars")
    min_line_lenght = cv2.getTrackbarPos("minLineLength", "Trackbars")
    max_line_gap = cv2.getTrackbarPos("maxLineGap", "Trackbars")
    info = np.float32([rho, threshold, min_line_lenght, max_line_gap])
    return info


def draw_circles(img, points):
    for x in range(4):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED)
    return img


# TEMPORAL
def stop():
    set_robot_speed(0)
    set_robot_direction("detenido", 0)
    print("Stop - Frenando")


def go_left(speed):
    set_robot_speed(speed)
    set_robot_direction("izquierda", 1)
    print("Left - Hacia la izquierda")


def go_right(speed):
    set_robot_speed(speed)
    set_robot_direction("derecha", 2)
    print("Right - Hacia la derecha")


def go_straigth(speed):
    set_robot_speed(speed)
    set_robot_direction("recto", 3)
    print("Forward - Hacia delante")
