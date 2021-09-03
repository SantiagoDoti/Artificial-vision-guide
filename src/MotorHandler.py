from gpiozero import Robot
import RPi.GPIO as GPIO
import Utils

# L298N pines <> GPIO pines
IN1, IN2, IN3, IN4, EN = 27, 22, 23, 24, 25

mr_robot = Robot(left=(IN1, IN2), right=(IN3, IN4))

def go_left(speed, image):
    mr_robot.left(speed)
    Utils.print_info_text(image, 2, speed)
    print("Left - Hacia la izquierda")


def go_right(speed, image):
    mr_robot.right(speed)
    Utils.print_info_text(image, 3, speed)
    print("Right - Hacia la derecha")


def go_straigth(speed, image):
    mr_robot.forward()
    Utils.print_info_text(image, 1, speed)
    print("Forward - Hacia delante")


def stop(image):
    mr_robot.stop()
    Utils.print_info_text(image, 0, 0)
    print("Stop - Frenando")
    
    
def shutdown_motor():
    mr_robot.stop()
