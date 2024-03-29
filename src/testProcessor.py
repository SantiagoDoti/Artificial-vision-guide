import socket
import pickle
import cv2
import time
import struct
import imageProcessor
from threading import Thread

# from motorHandler import MotorHandler

# L298N pines <> GPIO pines
IN1, IN2, IN3, IN4, EN = 27, 22, 23, 24, 25
# motor_handler = MotorHandler(IN1, IN2, IN3, IN4, EN)
allow_guide = True

# Posiciones iniciales de los tracbarks de los warped points para visualizarlos en pantalla
# initial_trackbar_vals = [186, 161, 57, 262]
# Utils.initializeTrackbars(initial_trackbar_vals)

# Video pregrabado
# root_path = os.path.abspath(os.path.dirname(__file__))
# video_path = os.path.join(root_path, "../tests/testYellowWithe.mp4")
# video = cv2.VideoCapture(video_path)

# Video en directo desde CUALQUIER webcam
video = cv2.VideoCapture(0)
time.sleep(0.1)


# Creamos el socket para luego enviarle las imágenes
def create_server_socket():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    print("HOST IP: ", host_ip)
    port = 9999
    socket_address = (host_ip, port)
    server_socket.bind(socket_address)

    server_socket.listen(5)
    print("ESCUCHANDO EN: ", socket_address)

    client_socket, addr = server_socket.accept()
    print('OBTENIENDO CONEXIÓN DESDE: ', addr)

    return server_socket, client_socket


def interpret_user_input():
    global allow_guide
    user_input = input("Escriba el comando deseado: ")
    if user_input.lower() == "r":
        print(" ----------- DIRECCION DEL ROBOT HABILITADA ----------- ")
        allow_guide = True
    elif user_input.lower() == "s":
        print(" ----------- DIRECCION DEL ROBOT DESHABILITADA ----------- ")
        allow_guide = False


def process_common_camera_video():
    while True:
        flag, frame = video.read()

        # Loop video pregrabado
        # if not flag:
        #     video = cv2.VideoCapture(video_path)
        #     continue

        if allow_guide:
            car_offset, left_curve, right_curve, frame_processed = imageProcessor.process_image(frame)
            if frame_processed is not None:
                frame = frame_processed
            # if car_offset is not None:
            #     motor_handler.guide_robot(car_offset)

        # Enviamos el video procesado a traves del socket
        if client_socket:
            a = pickle.dumps(frame)
            message = struct.pack("Q", len(a)) + a
            client_socket.sendall(message)

        # cv2.imshow("Imagen con curvatura y desplazamiento", final_image)

        if cv2.waitKey(10) == 27:
            break

    # motor_handler.stop()
    server_socket.close()
    video.release()
    cv2.destroyAllWindows()


server_socket, client_socket = create_server_socket()
process_common_camera_video()
