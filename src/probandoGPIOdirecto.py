import RPi.GPIO as GPIO
from time import sleep

ENA, IN1, IN2, ENB, IN3, IN4, = 12, 27, 22, 26, 23, 24

initial_speed = 50

# Inicialización del motor A (izquierdo)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
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

GPIO.setwarnings(False)

def set_motorA_speed(val):
    Ap.ChangeDutyCycle(initial_speed + val)

def set_motorB_speed(val):
    Bp.ChangeDutyCycle(initial_speed + val)

def forward():
#     print("Hacia delante")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def finish():
    GPIO.cleanup()

def to_left():
    forward()
    set_motorA_speed(30)
    set_motorB_speed(-1 * initial_speed)

def to_right():
    forward()
    set_motorA_speed(-1 * initial_speed)
    set_motorB_speed(30)
    
def to_straight():
    forward()
    set_motorA_speed(30)
    set_motorB_speed(30)

def stop():
    set_motorA_speed(-30)
    set_motorB_speed(-30)

#to_left()
to_right()
sleep(3)
stop()
finish()

# while True:
#     forward()
#     numero = 0
# 
#     if numero == 0:
#         speed_right = -1 * initial_speed
#         speed_left = -1 * initial_speed
#     elif numero > -5 and numero < 5:
#         speed_right = 30
#         speed_left = 30
#     elif numero > 5:
#         speed_right = 30
#         speed_left = -1 * initial_speed
#     elif numero < -5:
#         speed_left = 30
#         speed_right = -1 * initial_speed
# 
# #     speed_right = 0
# #     speed_left = 30
# 
#     set_motorA_speed(speed_right)
#     print("Velocidad derecha: ",speed_right)
#     set_motorB_speed(speed_left)
#     print("Velocidad izquierda: ", speed_left)
#     sleep(2)
# finish()


