import socket
import pickle
import cv2
import time
import struct
import imageProcessor
import Utils
from motorHandler import MotorHandler
from picamera.array import PiRGBArray
from picamera import PiCamera

# L298N pines <> GPIO pines
IN1, IN2, IN3, IN4, EN = 27, 22, 23, 24, 25
motor_handler = MotorHandler(IN1, IN2, IN3, IN4, EN)
allow_guide = True


# Video en vivo de la Raspberry Pi Camera
def setup_raspberry_camera():
    camera = PiCamera()
    camera.resolution = (720, 480)
    camera.framerate = 32
    raw_capture = PiRGBArray(camera, size=(720, 480))
    time.sleep(0.1)
    return camera, raw_capture


# Creamos el socket para luego enviarle las imágenes
def create_server_socket():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = "192.168.1.70"    # IP de la Raspberry Pi en la red local
    print("HOST IP: ", host_ip)
    port = 9999
    socket_address = (host_ip, port)
    server_socket.bind(socket_address)

    server_socket.listen(5)
    print("ESCUCHANDO EN: ", socket_address)

    client_socket, addr = server_socket.accept()
    print('OBTENIENDO CONEXIÓN DESDE: ', addr)

    return server_socket, client_socket


def shutdown_secure():
    motor_handler.stop()
    raspberry_server_socket.close()
    cv2.destroyAllWindows()


def interpret_user_input():
    global allow_guide
    user_input = input("Escriba el comando deseado: ")
    if user_input.lower() == "r":
        print(" ----------- DIRECCION DEL ROBOT HABILITADA ----------- ")
        allow_guide = True
    elif user_input.lower() == "s":
        print(" ----------- DIRECCION DEL ROBOT DESHABILITADA ----------- ")
        allow_guide = False


def process_raspberry_video():
    for frame in raspberry_pi_camera.capture_continuous(raspberry_raw_capture, format="bgr", use_video_port=True):
        image = frame.array

        if allow_guide:
            car_offset, frame_processed = imageProcessor.process_image(image)
            if frame_processed is not None:
                image = frame_processed
                motor_handler.guide_robot(car_offset)

        # Enviamos el video procesado a traves del socket
        if pc_client_socket:
            a = pickle.dumps(image)
            message = struct.pack("Q", len(a)) + a
            pc_client_socket.sendall(message)

        # cv2.imshow("Imagen con curvatura y desplazamiento", final_image)

        raspberry_raw_capture.truncate(0)

        if cv2.waitKey(10) == 27:
            break
    shutdown_secure()


try:
    raspberry_server_socket, pc_client_socket = create_server_socket()
    raspberry_pi_camera, raspberry_raw_capture = setup_raspberry_camera()
    process_raspberry_video()
except KeyboardInterrupt:
    shutdown_secure()


