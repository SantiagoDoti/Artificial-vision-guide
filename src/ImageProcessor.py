import numpy as np
import cv2

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


def make_points(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.001, 0
    y1 = int(image.shape[0])
    y2 = int(y1*3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
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


def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    mask = cv2.inRange(img, low_white, up_white)
    return cv2.Canny(mask, 75, 150)


def warp_image(img, points, width, height):
    points1 = np.float32(points)
    points2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(points1, points2)
    img_warped = cv2.warpPerspective(img, matrix, (width, height))
    return img_warped


def setup_image(frame, houghInfo):
    # Aplicamos varios filtros a la imagen y luego detectamos los bordes
    processed_image = process_image(frame)

    # Recortamos la imagen (ROI) para reducir el ruido y los falsos positivos
    cropped_image = region_of_interest(processed_image)

    # Creamos las lineas sobre la imagen negra cortada
    if houghInfo is None:
        lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=50, lines=np.array([]), minLineLength=60, maxLineGap=40)
    else:
        lines = cv2.HoughLinesP(cropped_image, rho=houghInfo[0], theta=np.pi/180, threshold=int(houghInfo[1]), lines=np.array([]), minLineLength=houghInfo[2], maxLineGap=houghInfo[3])

    averaged_lines = average_slope_intercept(frame, lines)
    lines_image = draw_lane_lines(frame, averaged_lines)

    # Combinamos la imagen negra con las lineas dibujadas contra la del video para obtener la imagen final
    lines_lane_image = cv2.addWeighted(frame, 0.8, lines_image, 1, 1)

    final_image = draw_heading_line(lines_lane_image, 90)

    return final_image
