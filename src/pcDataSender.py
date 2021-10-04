import socket


def tcp_server_wait(numofclientwait, port):
    global s2
    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s2.bind(('', port))
    s2.listen(numofclientwait)


def tcp_server_next():
    global s
    s = s2.accept()[0]


def tcp_write(D):
    s.send(D + b'\r')
    return


def tcp_close():
    s.close()
    s2.close()
    return


tcp_server_wait(5, 17098)  # Se envia la información por un puerto diferente
tcp_server_next()
