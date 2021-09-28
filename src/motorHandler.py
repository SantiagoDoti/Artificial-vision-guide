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
    mr_robot.left(speed)
    print("Moviéndose hacia la izquierda [{:0.2f}] ".format(center_offset))
    # time.sleep(0.2)
    # driving = False


def go_right(speed, center_offset):
    # global driving
    # driving = True
    mr_robot.right(speed)
    print("Moviéndose hacia la derecha [{:0.2f}] ".format(center_offset))
    # time.sleep(0.2)
    # driving = False


def go_straigth(speed, center_offset):
    # global driving
    # driving = True
    mr_robot.forward()
    print("Moviéndose hacia delante (recto) [{:0.2f}] ".format(center_offset))
    # time.sleep(0.2)
    # driving = False


def stop():
    # global driving
    # driving = True
    mr_robot.stop()
    print("Deteniendo el robot")
    # time.sleep(0.2)
    # driving = False


def shutdown_motor():
    # global driving
    mr_robot.stop()
    print("Apagando el motor del robot")
