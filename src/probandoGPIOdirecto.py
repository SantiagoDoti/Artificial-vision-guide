import RPi.GPIO as GPIO
import time

ENA, IN1, IN2, ENB, IN3, IN4, = 12, 27, 22, 20, 23, 24

initial_speed = 30

# Inicialización del motor A (izquierdo)
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.output(IN1, GPIO.LOW)
GPIO.output(IN2, GPIO.LOW)
Ap = GPIO.PWM(ENA, 1000)
Ap.start(initial_speed)

# Inicialización del motor B (derecho)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)
GPIO.output(IN3, GPIO.LOW)
GPIO.output(IN4, GPIO.LOW)
Bp = GPIO.PWM(ENB, 1000)
Bp.start(initial_speed)


def set_motorA_speed(val):
    Ap.ChangeDutyCycle(initial_speed + val)


def set_motorB_speed(val):
    Bp.ChangeDutyCycle(initial_speed + val)


def forward():
    print("Hacia delante")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)


def finish():
    GPIO.cleanup()


speed_right, speed_left, setback = 0, 0, 0

for X in [-10, -5, -1, 5, 10]:
    numero = X

    if numero > 0:
        speed_right = 10
        speed_left = -1 * initial_speed
    elif numero < 0:
        speed_left = 10
        speed_right = -1 * initial_speed

    set_motorA_speed(speed_right)
    set_motorB_speed(speed_left)
    forward()
finish()
