from gpiozero import Robot
import RPi.GPIO as GPIO
import Utils
import time

# L298N pines <> GPIO pines
IN1, IN2, IN3, IN4, EN = 27, 22, 23, 24, 25

# mr_robot = Robot(left=(IN1, IN2), right=(IN3, IN4))
# Definimos un rango de seguridad, el robot debe permanecer como máximo desplazado 5 cm a la derecha o a la izquierda
safety_zone_range = 0.05
# driving = False

GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.out)
GPIO.setup(IN2, GPIO.out)
GPIO.setup(IN3, GPIO.out)
GPIO.setup(IN4, GPIO.out)
GPIO.setup(EN, GPIO.out)

# Control del motor de dirección
GPIO.output(IN1, GPIO.HIGH)
GPIO.output(IN2, GPIO.LOW)
steering = GPIO.PWN(EN, 1000)
steering.stop()

# Control de motores de aceleración
GPIO.output(IN3, GPIO.HIGH)
GPIO.output(IN4, GPIO.LOW)
throttle = GPIO.PWN(EN, 1000)
throttle.stop()

time.sleep(1)

throttle.start(25)
steering.start(100)

time.sleep(3)

throttle.stop()
steering.stop()


def guide_robot_sides(center_offset):
    if center_offset > safety_zone_range:
        go_left(0.3, center_offset)
    elif center_offset < (- safety_zone_range):
        go_right(0.3, center_offset)
    else:
        go_straigth(0.3, center_offset)


def go_left(speed, center_offset):
    # global driving
    # driving = True
    # mr_robot.left(speed)
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    steering.start(100)
    print("Moviéndose hacia la izquierda [{:0.2f}] ".format(center_offset))
    # time.sleep(0.2)
    # driving = False


def go_right(speed, center_offset):
    # global driving
    # driving = True
    # mr_robot.right(speed)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    steering.start(100)
    print("Moviéndose hacia la derecha [{:0.2f}] ".format(center_offset))
    # time.sleep(0.2)
    # driving = False


def go_straigth(speed, center_offset):
    # global driving
    # driving = True
    # mr_robot.forward()
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    print("Moviéndose hacia delante (recto) [{:0.2f}] ".format(center_offset))
    # time.sleep(0.2)
    # driving = False


def stop():
    # global driving
    # driving = True
    # mr_robot.stop()
    print("Deteniendo el robot")
    # time.sleep(0.2)
    # driving = False


def shutdown_motor():
    # global driving
    # mr_robot.stop()
    print("Apagando el motor del robot")
