import socket
import pickle
import cv2
import time
import struct
import imageProcessor

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

while True:
    flag, frame = video.read()

    # Loop video pregrabado
    # if not flag:
    #     video = cv2.VideoCapture(video_path)
    #     continue

    final_image = imageProcessor.process_image(frame)

    # Enviamos el video procesado a traves del socket
    if client_socket:
        a = pickle.dumps(final_image)
        message = struct.pack("Q", len(a)) + a
        client_socket.sendall(message)

    # cv2.imshow("Imagen con curvatura y desplazamiento", final_image)

    if cv2.waitKey(10) == 27:
        break

server_socket.close()
video.release()
cv2.destroyAllWindows()

