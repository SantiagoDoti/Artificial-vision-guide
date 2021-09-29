from gpiozero import Robot
import RPi.GPIO as GPIO
import Utils
import time


# # mr_robot = Robot(left=(IN1, IN2), right=(IN3, IN4))
# # Definimos un rango de seguridad, el robot debe permanecer como máximo desplazado 5 cm a la derecha o a la izquierda
# safety_zone_range = 0.05
# # driving = False
#
# # L298N pines <> GPIO pines
# IN1, IN2, IN3, IN4, EN = 27, 22, 23, 24, 25


class MotorHandler:

    def __init__(self, IN1, IN2, IN3, IN4, EN):
        self.IN1, self.IN2, self.IN3, self.IN4, self.EN = IN1, IN2, IN3, IN4, EN

        self.safety_zone_range = 0.05

        self.driving = False

    def setup_motors(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.IN1, GPIO.OUT)
        GPIO.setup(self.IN2, GPIO.OUT)
        GPIO.setup(self.IN3, GPIO.OUT)
        GPIO.setup(self.IN4, GPIO.OUT)
        GPIO.setup(self.EN, GPIO.OUT)
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)

        p = GPIO.PWM(self.EN, 1000)

        p.start(25)

    def go_left(self, speed, center_offset):
        print("Moviéndose hacia la izquierda [{:0.2f}] ".format(center_offset))
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)
        time.sleep(0.5)
        self.driving = False

    def go_right(self, speed, center_offset):
        print("Moviéndose hacia la derecha [{:0.2f}] ".format(center_offset))
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        time.sleep(0.5)
        self.driving = False

    def go_straigth(self, speed, center_offset):
        print("Moviéndose hacia delante (recto) [{:0.2f}] ".format(center_offset))
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        time.sleep(1)
        self.driving = False

    def stop(self):
        print("Deteniendo el robot")
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)
        self.driving = False

    def guide_robot(self, center_offset):
        if not self.driving:
            self.driving = True
            if center_offset > self.safety_zone_range:
                self.go_left(0.3, center_offset)
            elif center_offset < (- self.safety_zone_range):
                self.go_right(0.3, center_offset)
            else:
                self.go_straigth(0.3, center_offset)

    # def shutdown_motor(self):
    #     # global driving
    #     # mr_robot.stop()
    #     print("Apagando el motor del robot")


def guide_robot_sides(center_offset):
    # L298N pines <> GPIO pines
    IN1, IN2, IN3, IN4, EN = 27, 22, 23, 24, 25

    motor_handler = MotorHandler(IN1, IN2, IN3, IN4, EN)
    motor_handler.guide_robot(center_offset)

