import io
import socket
import struct
import time
from picamera import PiCamera


# Creamos el socket para luego enviarle las imágenes en crudo
def create_socket():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('192.168.1.33', 8000))
    connection = client_socket.makefile('wb')
    return client_socket, connection


def capture_raspberry_video(raspberry_connection):
    try:
        with picamera.Picamera() as raspberry_pi_camera:
            raspberry_pi_camera.resolution = (720, 480)
            raspberry_pi_camera.framerate = 32
            time.sleep(1)
            start = time.time()
            stream = io.BytesIO()

            # Enviamos el video en vivo en formato JPEG
            for frame in raspberry_pi_camera.capture_continuous(stream, 'jpeg', use_video_port=True):
                raspberry_connection.write(struct.pack('<L', stream.tell()))
                raspberry_connection.flush()
                stream.seek(0)
                raspberry_connection.write(stream.read())
                stream.seek(0)
                stream.truncate()

            raspberry_connection.write(struct.pack("<L, 0"))
    except socket.error as e:
        print(e)
    finally:
        raspberry_connection.close()
        raspberry_socket.close()
        print("Conexión cerrada")


raspberry_socket, raspberry_connection = create_socket()
capture_raspberry_video(raspberry_connection)
