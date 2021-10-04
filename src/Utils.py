import cv2
import numpy as np


def nothing(a):
    pass


# Inicializar trackbars de posiciones de los puntos extremos del ROI
def initializeTrackbars(initial_tracbar_vals):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)

    cv2.createTrackbar("Width Top", "Trackbars", initial_tracbar_vals[0], wT // 2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", initial_tracbar_vals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", initial_tracbar_vals[2], wT // 2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", initial_tracbar_vals[3], hT, nothing)


# Trackbars para modificar las posiciones de los puntos extremos del ROI
def valTrackbars():
    width_top = cv2.getTrackbarPos("Width Top", "Trackbars")
    height_top = cv2.getTrackbarPos("Height Top", "Trackbars")
    width_bottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    height_bottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(width_top, height_top), (wT - width_top, height_top),
                         (width_bottom, height_bottom), (wT - width_bottom, height_bottom)])
    return points


# Inicializar trackabars de parámetros de Hough
def initializeTrackbars2(initial_tracbar_vals):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)

    cv2.createTrackbar("Rho", "Trackbars", initial_tracbar_vals[0], wT // 2, nothing)
    cv2.createTrackbar("Threshold", "Trackbars", initial_tracbar_vals[1], hT, nothing)
    cv2.createTrackbar("minLineLength", "Trackbars", initial_tracbar_vals[2], wT // 2, nothing)
    cv2.createTrackbar("maxLineGap", "Trackbars", initial_tracbar_vals[3], hT, nothing)


# Trackbars para modificar los parámetros de Hough
def valTrackbars2():
    rho = cv2.getTrackbarPos("Rho", "Trackbars")
    threshold = cv2.getTrackbarPos("Threshold", "Trackbars")
    min_line_lenght = cv2.getTrackbarPos("minLineLength", "Trackbars")
    max_line_gap = cv2.getTrackbarPos("maxLineGap", "Trackbars")
    info = np.float32([rho, threshold, min_line_lenght, max_line_gap])
    return info


# Dibujar círculos sobre puntos específicos
def draw_circles(img, points):
    for x in range(4):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED)
    return img


# TEMPORAL
robot_direction = 3  # recto (straigth)
robot_direction_text = "detenido"
robot_speed = 0
wT, hT = 640, 480
# wT, hT = 960, 540
# wT, hT = 1280, 720

