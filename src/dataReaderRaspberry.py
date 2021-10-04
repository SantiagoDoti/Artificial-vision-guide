import socket


def tcp_connect(host_ip, port):
    global s
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host_ip, port))
    return


def tcp_read():
    a = ''
    b = b''
    while a != b'\r':
        a = s.recv(1)
        b = b + a
    return b


def tcp_close():
    s.close()
    return


tcp_connect('192.168.1.33', 17098)  # IP de la PC en la red local desde donde recibimos instrucciones
