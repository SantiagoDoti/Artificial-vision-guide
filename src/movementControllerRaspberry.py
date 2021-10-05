import socket
from time import sleep
import RPi.GPIO as GPIO

# Definimos un rango de seguridad, el robot debe permanecer como máximo desplazado 5 cm a la derecha o a la izquierda
safety_zone_range = 0.10

def finish():
    GPIO.cleanup()

def set_motorA_speed(val):
    Ap.ChangeDutyCycle(initial_speed + val)

def set_motorB_speed(val):
    Bp.ChangeDutyCycle(initial_speed + val)

def forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def to_left():
    forward()
    set_motorA_speed(30)
    set_motorB_speed(-1 * initial_speed)

def to_right():
    forward()
    set_motorA_speed(-1 * initial_speed)
    set_motorB_speed(30)
    
def to_straight():
    set_motorA_speed(30)
    set_motorB_speed(30)
    forward()

def stop():
    set_motorA_speed(-30)
    set_motorB_speed(-30)

def client_program():
    client_socket = socket.socket()
    client_socket.connect(('192.168.1.33', 5500))
    
    driving = False
    speed_motorA, speed_motorB = 0, 0

    while True:
        data = client_socket.recv(1024).decode()

        try:
            robot_offset = float(data)
            print('Desplaz.: ' + '{:03.2f}'.format(abs(robot_offset)) + 'm')
        except ValueError:
            robot_offset = None
#             print("Valor fuera del rango, error al parsearlo")


        # left_curve = dat[1]
        # right_curve = dat[2]

        # print('Radio izquierdo: ' + '{:04.0f}'.format(left_curve) + ' m')
        # print('Radio derecho: ' + '{:04.0f}'.format(right_curve) + ' m')

        if robot_offset is not None and driving is False:
            driving = True
            if robot_offset == 0:  # Detenerse
                print("DETENERSE")
                stop()
            elif robot_offset > (- safety_zone_range) and robot_offset < safety_zone_range:
                print("DERECHO")
                to_straight()
            elif robot_offset > safety_zone_range:  # Moverse a la izquierda
                print("IZQUIERDA")
                to_left()
            elif robot_offset < (- safety_zone_range):  # Moverse a la derecha
                print("DERECHA")
                to_right()

#             set_motorA_speed(speed_motorA)
#             set_motorB_speed(speed_motorB)
#             forward()
            sleep(1)
            driving = False

    client_socket.close()

if __name__ == '__main__':
    GPIO.setwarnings(False)

    # L298N pines <> GPIO pines
    ENA, IN1, IN2, ENB, IN3, IN4, = 12, 27, 22, 26, 23, 24

    initial_speed = 60  # Velocidad inicial [0 - 100]

    # Inicialización del motor A (izquierdo)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(ENA, GPIO.OUT)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    Ap = GPIO.PWM(ENA, 500)
    Ap.start(initial_speed)

    # Inicialización del motor B (derecho)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)
    GPIO.setup(ENB, GPIO.OUT)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    Bp = GPIO.PWM(ENB, 500)
    Bp.start(initial_speed)

    client_program()
