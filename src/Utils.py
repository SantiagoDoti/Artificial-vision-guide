import cv2
import numpy as np

robot_direction_text = "detenido"
robot_speed = 0
wT, hT = 960, 540


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


def nothing(a):
    pass


def initializeTrackbars(intialTracbarVals):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)

    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0], wT // 2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2], wT // 2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)


def valTrackbars():
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                         (widthBottom, heightBottom), (wT-widthBottom, heightBottom)])
    return points


def initializeTrackbars2(intialTracbarVals):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)

    cv2.createTrackbar("Rho", "Trackbars", intialTracbarVals[0], wT // 2, nothing)
    cv2.createTrackbar("Threshold", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("minLineLength", "Trackbars", intialTracbarVals[2], wT // 2, nothing)
    cv2.createTrackbar("maxLineGap", "Trackbars", intialTracbarVals[3], hT, nothing)


def valTrackbars2():
    rho = cv2.getTrackbarPos("Rho", "Trackbars")
    threshold = cv2.getTrackbarPos("Threshold", "Trackbars")
    minLineLenght = cv2.getTrackbarPos("minLineLength", "Trackbars")
    maxLineGap = cv2.getTrackbarPos("maxLineGap", "Trackbars")
    info = np.float32([rho, threshold, minLineLenght, maxLineGap])
    return info


def draw_circles(img, points):
    for x in range(4):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED)
    return img



