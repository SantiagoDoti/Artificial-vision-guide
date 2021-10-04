import RPi.GPIO as GPIO

# L298N pines <> GPIO pines
ENA, IN1, IN2, ENB, IN3, IN4, = 12, 27, 22, 26, 23, 24

initial_speed = 30  # Velocidad inicial [0 - 100]

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
    # print("Hacia delante")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)


def finish():
    GPIO.cleanup()
