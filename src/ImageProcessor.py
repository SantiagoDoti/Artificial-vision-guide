import math
# import MotorHandler
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import Utils

scale_ratio = 1

# Variables globales
prev_leftx = None
prev_lefty = None
prev_rightx = None
prev_righty = None
prev_left_fit = []
prev_right_fit = []

prev_leftx2 = None
prev_lefty2 = None
prev_rightx2 = None
prev_righty2 = None
prev_left_fit2 = []
prev_right_fit2 = []


# trapeze1 => testWhiteRight.mp4
# trapeze2 => testYellowWithe.mp4
# trapeze3 => Raspberry Pi Camera

class ImageProcessor:

    def __init__(self, orig_frame):

        self.orig_frame = orig_frame

        # Mantenemos la imagen con las lineas de carril dibujadas
        self.lane_line_markings = None

        # Mantenemos la imagen despues de la transformación de perspectiva
        self.warped_frame = None
        self.transformation_matrix = None
        self.inv_transformation_matrix = None

        self.orig_image_size = self.orig_frame.shape[::-1][1:]

        height = self.orig_frame.shape[0]
        # height = self.orig_image_size[0]
        width = self.orig_frame.shape[1]
        # width = self.orig_image_size[1]
        self.width = width
        self.height = height

        # testWhiteRight.mp4
        # self.roi_points = np.float32([
        #     (100, height), (450, 300), (500, 300), (900, height)
        # ])

        # self.padding = int(0.25 * width)
        # self.warped_points = np.float32([[192, 638], [492, 468], [788, 468], [1088, 638]])
        self.warped_points = np.float32([[492, 498], [788, 498], [192, 638], [1088, 638]])
        self.desired_roi_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        # self.desired_roi_points = np.float32([
        #     # (0, 0), (0, 300), (500, 300), (900, height)
        #     [self.padding, 0],  # Top-left corner
        #     [self.padding, self.orig_image_size[1]],  # Bottom-left corner
        #     [self.orig_image_size[0] - self.padding, self.orig_image_size[1]],  # Bottom-right corner
        #     [self.orig_image_size[0] - self.padding, 0]  # Top-right corner
        # ])

        # testYellowWithe.mp4
        self.roi_points = np.float32([
            (255, 625), (533, 472), (742, 472), (1025, 625)
        ])

        # Raspberry Pi Camera
        # self.roi_points = np.float32([
        #     (0, height), (120, height / 3), (600, height / 3), (width, height)
        # ])

        # Histograma que muestra los picos de pixeles blancos en la detección de carriles
        self.histogram = None

        # Parámetros de las ventanas deslizantes
        self.no_of_windows = 10
        self.margin = int((1 / 12) * width)  # Anchura de la ventana es +/- margen
        self.minpix = int((1 / 24) * width)  # Número mínimo de píxeles para recentrar la ventana

        # Líneas polinómicas de mejor ajuste para las líneas de carriles (izquierda y derecha)
        self.left_fit = None
        self.right_fit = None
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.ploty = None
        self.left_fitx = None
        self.right_fitx = None
        self.leftx = None
        self.rightx = None
        self.lefty = None
        self.righty = None

        # Parámetros de pixeles para las dimensiones X e Y
        # Suponemos que el carril tiene unos 30 mts de largo y 3.7 mts de ancho. Dado que la imagen se deformó para la
        # perspectiva eliminamos algunos píxeles.
        self.YM_PER_PIX = 30.0 / 720
        self.XM_PER_PIX = 3.7 / 700

        # Radios de curvatura y desplazamiento
        self.left_curvem = None
        self.right_curvem = None
        self.center_offset = None

        # def region_of_interest(self, img):
    #     height = img.shape[0]
    #     width = img.shape[1]
    #     triangle = [np.array([(150, height), (850, height), (450, 320)])]
    #     trapeze1 = [np.array([(0, height), (width / 2, height / 2), (width, height)], dtype=np.int32)]
    #     trapeze2 = [np.array([(255, 625), (533, 472), (742, 472), (1025, 625)], dtype=np.int32)]
    #     trapeze3 = [np.array([(0, height), (120, height / 3), (600, height / 3), (width, height)], dtype=np.int32)]
    #     mask = np.zeros_like(img)
    #     cv2.fillPoly(mask, trapeze1, 255)
    #     # cv2.imshow("ROI", mask)
    #     return cv2.bitwise_and(img, mask)

    # Filtro gaussiano especificando el tamaño del kernel
    def blur_gaussian(self, channel, ksize=3):
        return cv2.GaussianBlur(channel, (ksize, ksize), 0)

    def threshold(self, channel, thresh=(128, 255), thresh_type=cv2.THRESH_BINARY):
        # Si la intensidad del pixel es mayor a thresh[0], se convierte el valor
        # a blanco (255), en caso contrario a negro (0)
        return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)

    # Encuentra los bordes que están alineados vertical y horizontalmente en la imagen
    def sobel(self, img_channel, orient='x', sobel_kernel=3):
        if orient == 'x':
            sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, sobel_kernel)
        if orient == 'y':
            sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, sobel_kernel)

        return sobel

    # Retorna un array binario de dos dimensiones (máscara) en la que todos los píxeles son 0 o 1
    def binary_array(self, array, thresh, value=0):
        if value == 0:
            binary = np.ones_like(array)
        else:
            binary = np.zeros_like(array)
            value = 1

        binary[(array >= thresh[0]) & (array <= thresh[1])] = value

        return binary

    # Sobel edge detection
    def mag_thresh(self, image, sobel_kernel=3, thresh=(0, 255)):
        # Obtenemos la magnitud de los bordes que están alineados verticalmente y horizotalmente en la imagen
        sobelx = np.absolute(self.sobel(image, orient='x', sobel_kernel=sobel_kernel))
        sobely = np.absolute(self.sobel(image, orient='y', sobel_kernel=sobel_kernel))

        # Encontrar las areas de la imagen que tienen los cambios de intensidad de píxeles mas fuertes
        mag = np.sqrt(sobelx ** 2 + sobely ** 2)

        return self.binary_array(mag, thresh)

    def process_image(self, frame=None):

        if frame is None:
            frame = self.orig_frame

        hls_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        # cv2.imshow("HLS image", hls_image)

        # sxbinary es una matriz llena de intensidades de blancos (ceros) y negros (255)
        _, sxbinary = self.threshold(hls_image[:, :, 1], thresh=(120, 255))
        sxbinary = self.blur_gaussian(sxbinary, ksize=3)

        # Los unos van a estar en las celdas con los valores mas altos de la derivada Sobel
        # (bordes de linea de carril mas fuertes)
        sxbinary = self.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))

        # Generamos un umbral binario sobre el canal S (saturacion) para eliminar los colores apagados de la carretera
        # Tendra una imagen llena de valores de intensidad 0 (negro) y 255 (blanco). Aquellos > 5 se estableceran en
        # blanco mientras que todos los demas en negro
        s_channel = hls_image[:, :, 2]
        _, s_binary = self.threshold(s_channel, (5, 255))

        # Generamos un umbral binario sobre el canal rojo (R) del canal BGR del frame original ya que los colores de las
        # líneas de los carriles poseen valores de rojos puros: blanco (255, 255, 255) y amarillo (0, 255, 255)
        # Aquellos > 120 se estableceran en blanco mientras que todos los demas en negro
        _, r_thresh = self.threshold(frame[:, :, 2], thresh=(120, 255))

        # Aplicamos una operacion AND bit a bit para reducir el ruido y los pixeles que no parezcan ser colores sólidos
        rs_binary = cv2.bitwise_and(s_binary, r_thresh)
        # cv2.imshow("Combinacion entre S y R", rs_binary)

        # Combinamos las posibles lineas de carriles con las posibles bordes de lineas de carril
        self.lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(np.uint8))
        # cv2.imshow("IMAGEN FINAL", lane_line_markings)

        return self.lane_line_markings

    # Dibujamos el ROI sobre la imagen y lo visualizamos
    def plot_roi(self, frame=None, plot=False):
        if not plot:
            return

        if frame is None:
            frame = self.orig_frame.copy()

        this_image = cv2.polylines(frame, np.int32([self.roi_points]), True, (147, 20, 255), 3)

        while True:
            cv2.imshow("ROI image", this_image)

            if cv2.waitKey(0):
                break

        cv2.destroyAllWindows()

    # Calcula el histograma de concentración de pixeles blancos sobre la warped image
    def calculate_histogram(self, frame=None, plot=True):
        if frame is None:
            frame = self.warped_frame

        self.histogram = np.sum(frame[int(
            frame.shape[0] / 2):, :], axis=0)

        if plot:
            figure, (ax1, ax2) = plt.subplots(2, 1)
            figure.set_size_inches(10, 5)
            ax1.imshow(frame, cmap='gray')
            ax1.set_title("Warped binary frame")
            ax2.plot(self.histogram)
            ax2.set_title("Histograma de picos")
            plt.show()

        return self.histogram

    # Obtenemos el pico izquierdo y derecho del histograma. Devuelve la coordenada X de cada pico
    def histogram_peak(self):
        midpoint = int(self.histogram.shape[0] / 2)
        leftx_base = np.argmax(self.histogram[:midpoint])
        rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint

        return leftx_base, rightx_base

    # Calcula la curvatura de la ruta en metros
    def calculate_curvature(self, print_in_terminal=False):

        # Establecemos el valor Y donde queremos calcular la curvatura de la carretera
        y_eval = np.max(self.ploty)

        left_fit_cr = np.polyfit(self.lefty * self.YM_PER_PIX, self.leftx * self.XM_PER_PIX, 2)
        right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * self.XM_PER_PIX, 2)

        left_curvem = ((1 + (2 * left_fit_cr[0] * y_eval * self.YM_PER_PIX + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curvem = ((1 + (2 * right_fit_cr[0] * y_eval * self.YM_PER_PIX + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        if print_in_terminal:
            print(left_curvem, "m", right_curvem, "m")

        self.left_curvem = left_curvem
        self.right_curvem = right_curvem

        return left_curvem, right_curvem

    # Calcula la posición del vehículo con respecto al centro
    def calculate_car_position(self, print_in_terminal=False):

        # Asumimos que la cámara está centrada en la imagen, obtenemos la posición del coche en centimetros
        # car_location = self.orig_frame.shape[1] / 2
        # frame.shape[0]
        left_pos = self.left_fit[0] * (self.height ** 2) + self.left_fit[1] * self.height + self.left_fit[2]
        right_pos = self.right_fit[0] * (self.height ** 2) + self.right_fit[1] * self.height + self.right_fit[2]
        car_location = (left_pos + right_pos) / 2

        center_cam = self.width / 2
        offset = np.absolute(center_cam - car_location)
        center_offset = offset * self.XM_PER_PIX

        # cv2.circle(frame, (int(car_location), 0), 15, (0, 0, 255), cv2.FILLED)
        #
        # # Fijamos la coordenada X del fondo de la linea del carril
        # bottom_left = self.left_fit[0] * height ** 2 + self.left_fit[1] * height + self.left_fit[2]
        # bottom_right = self.right_fit[0] * height ** 2 + self.right_fit[1] * height + self.right_fit[2]
        #
        # center_lane = (bottom_right - bottom_left) / 2 + bottom_left
        # center_offset = (np.abs(car_location) - np.abs(center_lane)) * self.XM_PER_PIX * 100

        if print_in_terminal:
            print(str(center_offset) + "cm")

        self.center_offset = center_offset

        return center_offset

    # Mostramos la información de curvatura y desplazamiento sobre la imagen
    def display_curvature_offset(self, frame=None):

        if frame is None:
            image_copy = self.orig_frame.copy()
        else:
            image_copy = frame

        cv2.putText(image_copy, 'Radio de la curva: ' + str((self.left_curvem + self.right_curvem) / 2)[:7] + ' m',
                    (int((5 / 600) * self.width), int((20 / 338) * self.height)),
                    cv2.FONT_HERSHEY_SIMPLEX, (float((0.5 / 600) * self.width)),
                    (55, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_copy, 'Desplazamiento del centro: ' + str(self.center_offset)[:7] + ' cm',
                    (int((5 / 600) * self.width), int((40 / 338) * self.height)),
                    cv2.FONT_HERSHEY_SIMPLEX, (float((0.5 / 600) * self.width)),
                    (255, 255, 255), 2, cv2.LINE_AA)

        return image_copy

    # Superponemos las lineas de carril sobre el frame original
    def overlay_lane_lines(self, plot=False):

        # Generamos una imagen para dibujar las lineas sobre ella
        warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Convertimos los puntos X e Y en un formato para cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Dibujamos sobre la imagen blanca deformada
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Deformamos el espacio en blanco de vuelta al espacio de la imagen original, usando la matrix inversa
        newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (self.orig_frame.shape[1], self.orig_frame.shape[0]))

        # Combinamos el resultado con la imagen original
        result = cv2.addWeighted(self.orig_frame, 1, newwarp, 0.3, 0)

        if plot:
            figure, (ax1, ax2) = plt.subplots(2, 1)
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            ax1.set_title("Frame original")
            ax2.set_title("Frame original con superposición de carriles")
            plt.show()

        return result

    # Usa la linea de carril de la sliding window anterior para obtener los parámetros
    def get_lane_line_previous_window(self, left_fit, right_fit, plot=False):

        margin = self.margin

        # Encontrar las coordenadas X e Y de todos los pixeles no nulos
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Guardamos los índices de píxeles del carril izquierdo y derecho
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                          (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
                (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds

        # Obtenemos las ubicaciones de los pixeles de las lineas de los carriles (izquierdo y derecho)
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        global prev_leftx2
        global prev_lefty2
        global prev_rightx2
        global prev_righty2
        global prev_left_fit2
        global prev_right_fit2

        # Nos aseguramos que tenemos píxeles distintos de cero
        if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
            leftx = prev_leftx2
            lefty = prev_lefty2
            rightx = prev_rightx2
            righty = prev_righty2

        self.leftx = leftx
        self.rightx = rightx
        self.lefty = lefty
        self.righty = righty

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Sumamos los últimos coeficientes del polinomio
        prev_left_fit2.append(left_fit)
        prev_right_fit2.append(right_fit)

        # Calculamos la media movil
        if len(prev_left_fit2) > 10:
            prev_left_fit2.pop(0)
            prev_right_fit2.pop(0)
            left_fit = sum(prev_left_fit2) / len(prev_left_fit2)
            right_fit = sum(prev_right_fit2) / len(prev_right_fit2)

        self.left_fit = left_fit
        self.right_fit = right_fit

        prev_leftx2 = leftx
        prev_lefty2 = lefty
        prev_rightx2 = rightx
        prev_righty2 = righty

        # Creamos los valores X e Y para graficar sobre la imagen
        ploty = np.linspace(0, self.warped_frame.shape[0] - 1, self.warped_frame.shape[0])
        try:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            print("La función falló en ajustar alguna linea!")
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        self.ploty = ploty
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx

        if plot:
            out_img = np.dstack((self.warped_frame, self.warped_frame, self.warped_frame)) * 255
            window_img = np.zeros_like(out_img)

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Creamos un polígono para mostrar el area de busqueda y convertimos los puntos X e Y en un formato
            # utilizable para cv2.fillPoly()
            margin = self.margin
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Dibujamos el carril sobre la imagen blanca deformada
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(self.warped_frame, cmap='gray')
            ax3.imshow(result)
            ax3.plot(left_fitx, ploty, color='yellow')
            ax3.plot(right_fitx, ploty, color='yellow')
            ax1.set_title("Frame original")
            ax2.set_title("Warped frame")
            ax3.set_title("Warped Frame con ventanas de búsqueda")
            plt.show()

    # Obtenemos los índices de los pixeles de las líneas de carriles usando la técnica de ventanas deslizantes
    def get_lane_line_indices_sliding_windows(self, plot=False):

        margin = self.margin

        frame_sliding_window = self.warped_frame.copy()

        # Seteamos la altura de las sliding windows (ventanas deslizantes)
        window_height = int(self.warped_frame.shape[0] / self.no_of_windows)

        # Buscamos las coordenadas X e Y de todos los píxeles no nulos (blancos) en el frame
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Almacenamos los índices de los pixeles de los carriles (izquierda y derecha)
        left_lane_inds = []
        right_lane_inds = []

        # Posiciones actuales para los índices de los pixeles de cada sliding window, que seguiremos actualizando
        leftx_base, rightx_base = self.histogram_peak()
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Se atraviesa una ventana a la vez
        no_of_windows = self.no_of_windows

        for window in range(no_of_windows):

            # Identificamos los límites de la ventana en X e Y (derecha e izquierda inclusive)
            win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height
            win_y_high = self.warped_frame.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(frame_sliding_window, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (255, 255, 255), 2)
            cv2.rectangle(frame_sliding_window, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (255, 255, 255), 2)

            # Identificamos los píxeles no nulos en X e Y dentro de la ventana
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (
                                      nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (
                                       nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # Si encontramos > minpix pixeles, receentramos la siguiente ventana en la posición media
            minpix = self.minpix
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Ajustamos una curva polinómica de 2° orden a las coordenadas de los píxeles de los carriles
        left_fit = None
        right_fit = None

        global prev_leftx
        global prev_lefty
        global prev_rightx
        global prev_righty
        global prev_left_fit
        global prev_right_fit

        # Nos aseguramos de tener píxeles no nulos
        if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
            leftx = prev_leftx
            lefty = prev_lefty
            rightx = prev_rightx
            righty = prev_righty

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Sumamos los últimos coeficientes del polinomio
        prev_left_fit.append(left_fit)
        prev_right_fit.append(right_fit)

        # Calculamos la media móvil
        if len(prev_left_fit) > 10:
            prev_left_fit.pop(0)
            prev_right_fit.pop(0)
            left_fit = sum(prev_left_fit) / len(prev_left_fit)
            right_fit = sum(prev_right_fit) / len(prev_right_fit)

        self.left_fit = left_fit
        self.right_fit = right_fit

        prev_leftx = leftx
        prev_lefty = lefty
        prev_rightx = rightx
        prev_righty = righty

        if plot:
            # Creamos los valores X e Y para dibujar en la imagen
            ploty = np.linspace(0, frame_sliding_window.shape[0] - 1, frame_sliding_window.shape[0])
            try:
                left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
                right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            except TypeError:
                print("La función falló en ajustar alguna linea!")
                left_fitx = 1 * ploty ** 2 + 1 * ploty
                right_fitx = 1 * ploty ** 2 + 1 * ploty

            # Generamos una imagen para visualizar el resultado y le añadimos color a los píxeles de los carriles
            out_img = np.dstack((frame_sliding_window, frame_sliding_window, frame_sliding_window)) * 255
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            figure, (ax1, ax2, ax3) = plt.subplots(3, 1)
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(frame_sliding_window, cmap='gray')
            ax3.imshow(out_img)
            ax3.plot(left_fitx, ploty, color='yellow')
            ax3.plot(right_fitx, ploty, color='yellow')
            ax1.set_title("Frame original")
            ax2.set_title("Warped frame with sliding windows (ventana deslizante)")
            ax3.set_title("Lineas detectadas con sliding windows")
            plt.show()

        return self.left_fit, self.right_fit

    # Warp pespective
    def perspective_transform(self, frame=None, plot=None):

        if frame is None:
            frame = self.lane_line_markings

        # Calculamos la matrix de transformación
        self.transformation_matrix = cv2.getPerspectiveTransform(
            self.warped_points, self.desired_roi_points)

        # Calculamos la inversa de la matrix de transformación
        self.inv_transformation_matrix = cv2.getPerspectiveTransform(
            self.desired_roi_points, self.warped_points)

        # Realizamos la transformación con la matrix de transformación
        self.warped_frame = cv2.warpPerspective(
            frame, self.transformation_matrix, self.orig_image_size, flags=cv2.INTER_LINEAR)

        # Convertimos la imagen en binaria
        (tresh, binary_warped) = cv2.threshold(self.warped_frame, 127, 255, cv2.THRESH_BINARY)
        self.warped_frame = binary_warped

        return self.warped_frame

    def setup_warped_points_image(self, frame):
        points = Utils.valTrackbars()
        points_warped_image = Utils.draw_circles(frame, points)

        return points_warped_image


def main():
    root_path = os.path.abspath(os.path.dirname(__file__))
    # video_path = os.path.join(root_path, "../tests/testWhiteRight.mp4")
    video_path = os.path.join(root_path, "../tests/testYellowWithe.mp4")

    # image = cv2.imread('imagenRoadPPS.png')

    video = cv2.VideoCapture(video_path)

    # initialTrackbarVals = [492, 498, 192, 638]
    # Utils.initializeTrackbars(initialTrackbarVals)

    while True:
        flag, frame = video.read()

        # Loop video
        if not flag:
            video = cv2.VideoCapture(video_path)
            continue

        width = int(frame.shape[1] * scale_ratio)
        height = int(frame.shape[0] * scale_ratio)
        frame = cv2.resize(frame, (width, height))

        original_frame = frame.copy()

        image_processor = ImageProcessor(orig_frame=original_frame)

        lane_line_markings = image_processor.process_image()

        image_processor.plot_roi(plot=False)

        warped_image = image_processor.perspective_transform(plot=False)
        # warped_points = image_processor.setup_warped_points_image(frame=frame)

        image_processor.calculate_histogram(plot=False)

        left_fit, right_fit = image_processor.get_lane_line_indices_sliding_windows(plot=False)

        image_processor.get_lane_line_previous_window(left_fit, right_fit, plot=False)

        frame_lane_lines = image_processor.overlay_lane_lines(plot=False)

        image_processor.calculate_curvature(print_in_terminal=True)

        image_processor.calculate_car_position(print_in_terminal=True)

        frame_with_info = image_processor.display_curvature_offset(frame=frame_lane_lines)

        # cv2.imshow("Imagen original", lane_line_markings)
        # cv2.imshow("Imagen deformada", warped_image)
        # cv2.imshow("Imagen con puntos de deforme", warped_points)
        # cv2.imshow("Imagen con trayecto dibujado ", frame_lane_lines)
        cv2.imshow("Imagen con curvatura y desplazamiento", frame_with_info)

        if cv2.waitKey(10) == 27:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
