import cv2
import numpy as np


# Sobel edge detection
def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    # Obtenemos la magnitud de los bordes que están alineados verticalmente y horizotalmente en la imagen
    sobelx = np.absolute(sobel(image, orient='x', sobel_kernel=sobel_kernel))
    sobely = np.absolute(sobel(image, orient='y', sobel_kernel=sobel_kernel))

    # Encontrar las areas de la imagen que tienen los cambios de intensidad de píxeles mas fuertes
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    return binary_array(mag, thresh)


# Retorna un array binario de dos dimensiones (máscara) en la que todos los píxeles son 0 o 1
def binary_array(array, thresh, value=0):
    if value == 0:
        binary = np.ones_like(array)
    else:
        binary = np.zeros_like(array)
        value = 1
    binary[(array >= thresh[0]) & (array <= thresh[1])] = value

    return binary


def threshold(channel, thresh=(128, 255), thresh_type=cv2.THRESH_BINARY):
    # Si la intensidad del pixel es mayor a thresh[0], se convierte el valor
    # a blanco (255), en caso contrario a negro (0)
    return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)


# Encuentra los bordes que están alineados vertical y horizontalmente en la imagen
def sobel(img_channel, orient='x', sobel_kernel=3):
    if orient == 'x':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, sobel_kernel)

    return sobel


# Filtro gaussiano especificando el tamaño del kernel
def blur_gaussian(channel, ksize=3):
    return cv2.GaussianBlur(channel, (ksize, ksize), 0)
