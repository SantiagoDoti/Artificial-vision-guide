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


def draw_lines(img, lines):
    # Creamos otra imagen totalmente negra del mismo tamaño que la original para dibujar las lineas sobre ella
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


def setup_image(frame):
    # Aplicamos varios filtros a la imagen y luego detectamos los bordes
    processed_image = process_image(frame)

    # Recortamos la imagen (ROI) para reducir el ruido y los falsos positivos
    cropped_image = region_of_interest(processed_image)

    # Creamos las lineas sobre la imagen negra cortada
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=20, maxLineGap=40)
    lines_image = draw_lines(frame, lines)

    # Combinamos la imagen negra con las lineas dibujadas contra la del video para obtener la imagen final
    final_image = cv2.addWeighted(frame, 0.8, lines_image, 1, 1)

    return final_image
