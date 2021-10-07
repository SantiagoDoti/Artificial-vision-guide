import numpy as np
import cv2
import socket
import imageProcessor


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_name = socket.gethostname()
socket_address = ('192.168.1.33', 5500)
server_socket.bind(socket_address)

server_socket.listen(5)
print("ESCUCHANDO EN: ", socket_address)

conn, address = server_socket.accept()


def server_program(data):
    data = str(data)
    conn.send(data.encode())


class VideoStreaming(object):

    def __init__(self, host, port):
        self.server_socket = socket.socket()
        self.server_socket.bind((host, port))
        self.server_socket.listen(0)
        self.connection, self.client_address = self.server_socket.accept()
        self.connection = self.connection.makefile('rb')
        self.host_name = socket.gethostname()
        self.host_ip = socket.gethostbyname(self.host_name)
        self.streaming()

    def streaming(self):
        try:
            print("Host: ", self.host_name + " " + self.host_ip)
            print("Obteniendo conexión desde: ", self.client_address)
            print("Streaming..")
            print("[Apretar Q en cualquier momento para salir]")

            stream_bytes = b' '
            while True:
                stream_bytes += self.connection.read(1024)
                first_byte = stream_bytes.find(b'\xff\xd8')
                last_byte = stream_bytes.find(b'\xff\xd9')
                if first_byte != -1 and last_byte != -1:
                    jpg = stream_bytes[first_byte:last_byte + 2]
                    stream_bytes = stream_bytes[last_byte + 2:]
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    lane_image = np.copy(image)
                    car_offset, left_curve, right_curve, frame_processed = imageProcessor.process_image(lane_image)
                    if frame_processed is not None:
                        lane_image = frame_processed
                        server_program(car_offset)
                    else:
                        server_program(None)

                    cv2.imshow("Imagen con curvatura y desplazamiento", lane_image)

                    if cv2.waitKey(10) == 27:
                        break
        finally:
            self.connection.close()
            self.server_socket.close()


if __name__ == '__main__':
    # host, puerto
    host, port = "192.168.1.33", 8000
    VideoStreaming(host, port)

cv2.destroyAllWindows()

