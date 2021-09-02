import math

import numpy as np
import cv2
import Utils

# trapeze1 => testWhiteRight.mp4
# trapeze2 => testYellowWithe.mp4


def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    triangle = [np.array([(150, height), (850, height), (450, 320)])]
    trapeze1 = [np.array([(0, height), (width / 2, height / 2), (width, height)], dtype=np.int32)]
    trapeze2 = [np.array([(255, 625), (533, 472), (742, 472), (1025, 625)], dtype=np.int32)]
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, trapeze1, 255)
    return cv2.bitwise_and(img, mask)


def make_points(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.001, 0
    y1 = int(image.shape[0])
    y2 = int(y1 * 3 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    return np.array((left_line, right_line))


def draw_lane_lines(img, lines):
    # Creamos otra imagen totalmente negra del mismo tamaño que la original para dibujar las lineas sobre ella
    lines_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            try:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 10)
            except Exception:
                print(Exception.args)
    return lines_image


def draw_heading_line(img, steering_angle):
    heading_image = np.zeros_like(img)
    height, width = img.shape[0], img.shape[1]

    steering_angle_radian = steering_angle / 180.0 * np.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / np.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    heading_image = cv2.addWeighted(img, 0.8, heading_image, 1, 1)

    return heading_image


def compute_steering_angle(frame, lane_lines):
    if len(lane_lines) == 0:
        print("No hay lineas detectadas, no se hace nada")
        return -90

    height, widht = frame.shape[0], frame.shape[1]
    if len(lane_lines) == 1:
        print("Se detectó una sola linea, se procede a seguirla. %s" % lane_lines[0])
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = 0.02  # 0.0: auto apunta al centro, -0.03: auto centrado a la izquierda, +0.03: auto centrado a la derecha
        mid = int(widht / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # ángulo (en radianes) a la linea central
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # ángulo (en grados) a la linea central
    steering_angle = angle_to_mid_deg + 90

    print("El nuevo ángulo de dirección es: %s" % steering_angle)
    return steering_angle


def stabilize_steering_angle(current_steering_angle, new_steering_angle, amount_lane_lines,
                             max_angle_deviation_two_lines=5, max_angle_deviation_one_line=1):
    if amount_lane_lines == 2:
        max_angle_deviation = max_angle_deviation_two_lines
    else:
        max_angle_deviation = max_angle_deviation_one_line

    angle_deviation = new_steering_angle - current_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilize_steering_angle = int(current_steering_angle + max_angle_deviation
                                       * angle_deviation / abs(angle_deviation))
    else:
        stabilize_steering_angle = new_steering_angle

    print("Ángulo propuesto: %s. Ángulo estabilizado: %s" % (new_steering_angle, stabilize_steering_angle))
    return stabilize_steering_angle


# def process_image(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     img = cv2.GaussianBlur(img, (5, 5), 0)
#     mask = cv2.inRange(img, low_white, up_white)
#     return cv2.Canny(mask, 75, 150)

def process_image(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, 75, 150)
    return canny_image


def warp_image(img, points, width, height):
    points1 = np.float32(points)
    points2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(points1, points2)
    img_warped = cv2.warpPerspective(img, matrix, (width, height))
    return img_warped


def setup_edges_image(frame):
    # Aplicamos varios filtros a la imagen y luego detectamos los bordes
    processed_image = process_image(frame)

    # Recortamos la imagen (ROI) para reducir el ruido y los falsos positivos
    cropped_image = region_of_interest(processed_image)

    return cropped_image


def setup_lane_detection_image(frame, houghInfo):
    cropped_image = setup_edges_image(frame)

    # Creamos las lineas sobre la imagen negra cortada
    lines = cv2.HoughLinesP(cropped_image, rho=houghInfo[0], theta=np.pi / 180, threshold=int(houghInfo[1]),
                            lines=np.array([]), minLineLength=houghInfo[2], maxLineGap=houghInfo[3])

    # Hacemos un promedio entre las lineas para dibujar UNA sola si hay varias cercanas
    averaged_lines = average_slope_intercept(frame, lines)
    lines_image = draw_lane_lines(frame, averaged_lines)

    # Combinamos la imagen negra con las lineas dibujadas contra la del video para obtener la imagen final
    lines_lane_image = cv2.addWeighted(frame, 0.8, lines_image, 1, 1)

    # Dibujamos linea central de guia con ¿un angulo de centralización?
    if averaged_lines is not None and len(averaged_lines) != 0:
        new_steering_angle = compute_steering_angle(frame, averaged_lines)
        steering_angle = stabilize_steering_angle(90, new_steering_angle, len(averaged_lines))
        final_image = draw_heading_line(lines_lane_image, steering_angle)
        return final_image
    else:
        return lines_lane_image


def setup_warped_image(frame):
    height, width = frame.shape[0], frame.shape[1]
    points = Utils.valTrackbars()
    warped_image = warp_image(frame, points, width, height)

    return warped_image


def setup_warped_points_image(frame):
    points = Utils.valTrackbars()
    points_warped_image = Utils.draw_circles(frame, points)

    return points_warped_image
