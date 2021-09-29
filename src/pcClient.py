import cv2
import pickle
import socket
import struct
import imageProcessor


# Creamos el socket del cliente
def create_client_socket():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # host_ip = '192.168.1.70'  # Acá va la IP del host (server)
    host_ip = '192.168.1.33'
    port = 9999
    client_socket.connect((host_ip, port))
    data = b""
    payload_size = struct.calcsize("Q")
    # print("payload_size: {}".format(payload_size))
    return client_socket, payload_size, data


def process_video_received(data):
    while True:
        while len(data) < payload_size:
            data += pc_client_socket.recv(4096)

        # print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += pc_client_socket.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = pickle.loads(frame_data)
        # final_image = imageProcessor.process_image(frame)
        cv2.imshow("RECIBIENDO VIDEO", frame)

        if cv2.waitKey(10) == 27:
            break

    pc_client_socket.close()


pc_client_socket, payload_size, data_received = create_client_socket()
process_video_received(data_received)
