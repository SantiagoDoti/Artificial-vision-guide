from gpiozero import Robot
import RPi.GPIO as GPIO
import Utils

# L298N pines <> GPIO pines
IN1, IN2, IN3, IN4, EN = 27, 22, 23, 24, 25

mr_robot = Robot(left=(IN1, IN2), right=(IN3, IN4))


def go_backward(speed):
    mr_robot.backward(speed)
    Utils.set_robot_speed(speed)
    Utils.set_robot_direction_text("atras")
    print("Backward - Marcha atras")


def go_forward(speed):
    mr_robot.forward()
    Utils.set_robot_speed(speed)
    Utils.set_robot_direction_text("delante")
    print("Forward - Hacia delante")


def go_left(speed):
    mr_robot.left(speed)
    Utils.set_robot_speed(speed)
    Utils.set_robot_direction_text("izquierda")
    print("Left - Hacia la izquierda")


def go_right(speed):
    mr_robot.right(speed)
    Utils.set_robot_speed(speed)
    Utils.set_robot_direction_text("derecha")
    print("Right - Hacia la derecha")


def stop():
    mr_robot.stop()
    Utils.set_robot_speed(0)
    Utils.set_robot_direction_text("detenido")
    print("Stop - Frenando")
