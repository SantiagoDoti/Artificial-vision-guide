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
    points = np.float32([(width_top, height_top), (wT-width_top, height_top),
                         (width_bottom, height_bottom), (wT-width_bottom, height_bottom)])
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
robot_direction = 3     # recto (straigth)
robot_direction_text = "detenido"
robot_speed = 0
wT, hT = 640, 480
#wT, hT = 960, 540
#wT, hT = 1280, 720


# Dibujamos sobre la imagen información importante sobre el rumbo actual del robot y la velocidad
def print_info_text(image, direction_option, speed):
    cv2.rectangle(image, (10, 10), (310, 100), (0, 0, 255), 4)
    cv2.putText(image, "Rumbo: ", (18, 50), 2, 1, (0, 0, 255), 2, cv2.LINE_AA)
    directions = np.array(["detenido", "recto", "izquierda", "derecha"])
    cv2.putText(image, directions[direction_option], (150, 50), 3, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Velocidad: ", (18, 80), 2, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str(speed), (190, 80), 3, 1, (255, 255, 255), 2, cv2.LINE_AA)
    print_direction_arrow(image, direction_option)


def print_direction_arrow(image, direction_option):
    width, height = image.shape[0], image.shape[1]

    #     x1 = int(width / 2)
    #     y1 = height
    if direction_option == 0:
        pass
    if direction_option == 1:   # recto (straigth)
        pt1 = np.array([width / 2, height / 4 + 80])
        pt2 = np.array([width / 2, height / 4])
    elif direction_option == 2:     # izquierda (left)
        pt1 = np.array([width / 2 - 30, height / 4])
        pt2 = np.array([width / 2 - 80, height / 4])
    elif direction_option == 3:     # derecha (right)
        pt1 = np.array([width / 2 + 30, height / 4])
        pt2 = np.array([width / 2 + 80, height / 4])
    cv2.arrowedLine(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 0, 255), thickness=10, tipLength=1)


def stop(image):
    # set_robot_speed(0)
    # set_robot_direction("detenido", 0)
    print_info_text(image, 0, 0)
    print("Stop - Frenando")


def go_straigth(speed, image):
    # set_robot_speed(speed)
    # set_robot_direction("recto", 3)
    print_info_text(image, 1, speed)
    print("Forward - Hacia delante")


def go_left(speed, image):
    # set_robot_speed(speed)
    # set_robot_direction("izquierda", 1)
    print_info_text(image, 2, speed)
    print("Left - Hacia la izquierda")


def go_right(speed, image):
    # set_robot_speed(speed)
    # set_robot_direction("derecha", 2)
    print_info_text(image, 3, speed)
    print("Right - Hacia la derecha")
