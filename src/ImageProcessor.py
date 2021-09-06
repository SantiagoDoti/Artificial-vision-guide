import math
# import MotorHandler
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import Utils


# trapeze1 => testWhiteRight.mp4
# trapeze2 => testYellowWithe.mp4
# trapeze3 => Raspberry Pi Camera

class ImageProcessor:

    def __init__(self, orig_frame):

        self.orig_frame = orig_frame

        # Mantenemos la imagen con las lineas de carril dibujadas
        self.lane_line_image = None

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
        self.roi_points = np.float32([
            (100, height), (450, 300), (500, 300), (900, height)
        ])
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
        # self.roi_points = np.float32([
        #     (255, 625), (533, 472), (742, 472), (1025, 625)
        # ])

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


    def region_of_interest(self, img):
        height = img.shape[0]
        width = img.shape[1]
        triangle = [np.array([(150, height), (850, height), (450, 320)])]
        trapeze1 = [np.array([(0, height), (width / 2, height / 2), (width, height)], dtype=np.int32)]
        trapeze2 = [np.array([(255, 625), (533, 472), (742, 472), (1025, 625)], dtype=np.int32)]
        trapeze3 = [np.array([(0, height), (120, height / 3), (600, height / 3), (width, height)], dtype=np.int32)]
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, trapeze1, 255)
        # cv2.imshow("ROI", mask)
        return cv2.bitwise_and(img, mask)

    def make_points(self, image, line_parameters):
        try:
            slope, intercept = line_parameters
        except TypeError:
            slope, intercept = 0.001, 0
        y1 = int(image.shape[0])
        y2 = int(y1 * 3 / 5)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [[x1, y1, x2, y2]]

    def average_slope_intercept(self, image, lines):
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
        left_line = self.make_points(image, left_fit_average)
        right_line = self.make_points(image, right_fit_average)
        return np.array((left_line, right_line))

    def draw_lane_lines(self, img, lines):
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

    def draw_heading_line(self, img, steering_angle):
        heading_image = np.zeros_like(img)
        height, width = img.shape[0], img.shape[1]

        steering_angle_radian = steering_angle / 180.0 * np.pi
        x1 = int(width / 2)
        y1 = height
        x2 = int(x1 - height / 2 / np.tan(steering_angle_radian))
        y2 = int(height / 1.5)

        cv2.line(heading_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
        heading_image = cv2.addWeighted(img, 0.8, heading_image, 1, 1)

        return heading_image

    def compute_steering_angle(self, frame, lane_lines):
        if len(lane_lines) == 0:
            print("No hay lineas detectadas, no se hace nada")
            return 0

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

        # print("El nuevo ángulo de dirección es: %s" % steering_angle)
        return steering_angle

    def stabilize_steering_angle(self, current_steering_angle, new_steering_angle, amount_lane_lines,
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

    def drive(steering_angle, image):
        if 45 < steering_angle < 90:
            Utils.go_right(0.3, image)
        elif steering_angle == 90:
            Utils.go_straigth(0.5, image)
        elif 90 < steering_angle < 136:
            Utils.go_left(0.3, image)

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

    def process_image(self, img=None):
        # gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # canny_image = cv2.Canny(blur_image, 75, 150)
        # return canny_image

        hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
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
        _, r_thresh = self.threshold(img[:, :, 2], thresh=(120, 255))

        # Aplicamos una operacion AND bit a bit para reducir el ruido y los pixeles que no parezcan ser colores sólidos
        rs_binary = cv2.bitwise_and(s_binary, r_thresh)
        # cv2.imshow("Combinacion entre S y R", rs_binary)

        # Combinamos las posibles lineas de carriles con las posibles bordes de lineas de carril
        lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(np.uint8))
        cv2.imshow("IMAGEN FINAL", lane_line_markings)

        return lane_line_markings

    # def warp_image(self, img, points, width, height):
    #     points1 = np.float32(points)
    #     points2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    #     matrix = cv2.getPerspectiveTransform(points1, points2)
    #     img_warped = cv2.warpPerspective(img, matrix, (width, height))
    #     return img_warped

    # Dibujamos el ROI sobre la imagen y lo visualizamos
    def plot_roi(self, frame=None, plot=False):
        if plot == False:
            return

        if frame is None:
            frame = self.orig_frame.copy()

        this_image = cv2.polylines(frame, np.int32([
            self.roi_points
        ]), True, (147, 20, 255), 3)

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


    # Obtenemos los índices de los pixeles de las líneas de carriles usando la técnica de ventanas deslizantes
    def get_lane_line_indices_sliding_windows(self,left_fit, right_fit, plot=False):
        margin = self.margin

        # Coordenadas X e Y de todos los pixeles no nulos (blancos) en el frame
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Almacenamos esos índices
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                          (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                          (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds




    # Warp pespective
    def perspective_transform(self, frame=None, plot=None):

        if frame is None:
            frame = self.lane_line_image

        # Calculamos la matrix de transformación
        self.transformation_matrix = cv2.getPerspectiveTransform(
            self.warped_points, self.desired_roi_points)

        # Calculamos la inversa de la matrix de transformación
        # self.inv_transformation_matrix = cv2.getPerspectiveTransform(
        #     self.desired_roi_points, self.roi_points)

        # Realizamos la transformación con la matrix de transformación
        self.warped_frame = cv2.warpPerspective(
            frame, self.transformation_matrix, self.orig_image_size, flags=cv2.INTER_LINEAR)

        # Convertimos la imagen en binaria
        (tresh, binary_warped) = cv2.threshold(self.warped_frame, 127, 255, cv2.THRESH_BINARY)
        self.warped_frame = binary_warped

        # if plot:
        #     warped_copy = self.warped_frame.copy()
        #     warped_plot = cv2.polylines(warped_copy, np.int32([self.desired_roi_points]), True, (147, 20, 255), 3)
        #
        #     while True:
        #         cv2.imshow("Warped image", warped_plot)
        #
        #         if cv2.waitKey(0):
        #             break
        #
        #     cv2.destroyAllWindows()

        return self.warped_frame

    # def setup_edges_image(frame):
    #     # Aplicamos varios filtros a la imagen y luego detectamos los bordes
    #     processed_image = process_image(frame)
    #
    #     # Recortamos la imagen (ROI) para reducir el ruido y los falsos positivos
    #     cropped_image = region_of_interest(processed_image)
    #
    #     return cropped_image

    # def setup_lane_detection_image(frame, hough_parameters):
    #     cropped_image = setup_edges_image(frame)
    #
    #     # Creamos las lineas sobre la imagen negra cortada
    #     if hough_parameters is not None:
    #         lines = cv2.HoughLinesP(cropped_image, rho=hough_parameters[0], theta=np.pi / 180, threshold=int(hough_parameters[1]),
    #                             lines=np.array([]), minLineLength=hough_parameters[2], maxLineGap=hough_parameters[3])
    #     else:
    #         lines = cv2.HoughLinesP(cropped_image, rho=1, theta=np.pi / 180, threshold=50,
    #                             lines=np.array([]), minLineLength=40, maxLineGap=60)
    #
    #     # Hacemos un promedio entre las lineas para dibujar UNA sola si hay varias cercanas
    #     averaged_lines = average_slope_intercept(frame, lines)
    #     lines_image = draw_lane_lines(frame, averaged_lines)
    #
    #     # Combinamos la imagen negra con las lineas dibujadas contra la del video para obtener la imagen final
    #     lines_lane_image = cv2.addWeighted(frame, 0.8, lines_image, 1, 1)
    #
    #     # Dibujamos linea central de guia con ¿un angulo de centralización?
    #     if averaged_lines is not None and len(averaged_lines) != 0:
    #         new_steering_angle = compute_steering_angle(frame, averaged_lines)
    #         steering_angle = stabilize_steering_angle(90, new_steering_angle, len(averaged_lines))
    #         drive(steering_angle, lines_lane_image)
    #         final_image = draw_heading_line(lines_lane_image, steering_angle)
    #     else:
    #         final_image = lines_lane_image
    #
    #     return final_image

    # def setup_warped_image(self, frame):
    #     height, width = frame.shape[0], frame.shape[1]
    #     points = Utils.valTrackbars()
    #     warped_image = warp_image(frame, points, width, height)
    #
    #     return warped_image
    #

    def setup_warped_points_image(self, frame):
        points = Utils.valTrackbars()
        points_warped_image = Utils.draw_circles(frame, points)

        return points_warped_image


def main():
    root_path = os.path.abspath(os.path.dirname(__file__))
    # videoPath = os.path.join(rootPath, "../tests/testWhiteRight.mp4")
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

        image_processor = ImageProcessor(orig_frame=frame)
        # image_processor.plot_roi(plot=True)
        warped_image = image_processor.perspective_transform(frame=frame, plot=True)
        # warped_points = image_processor.setup_warped_points_image(frame=frame)
        image_processor.calculate_histogram(plot=True)

        cv2.imshow("Warped image", warped_image)
        # cv2.imshow("Warp points", warped_points)
        if cv2.waitKey(10) == 27:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
