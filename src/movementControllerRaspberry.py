from time import sleep
import motorHandlerRaspberry
import dataReaderRaspberry

m1 = motorHandlerRaspberry
# Definimos un rango de seguridad, el robot debe permanecer como máximo desplazado 5 cm a la derecha o a la izquierda
safety_zone_range = 0.05

dat = [0, 0, 0, 0, 0]
while True:
    c = 0
    D = dataReaderRaspberry.tcp_read()
    for b in D:
        dat[c] = b
        c += 1

        robot_offset = dat[0]
        left_curve = dat[1]
        right_curve = dat[2]

        print("[INFORMACIÓN RECIBIDA]")
        print('Desplaz.: ' + '{:03.2f}'.format(abs(robot_offset)) + 'm')
        print('Radio izquierdo: ' + '{:04.0f}'.format(left_curve) + ' m')
        print('Radio derecho: ' + '{:04.0f}'.format(right_curve) + ' m')

        speed_motorA, speed_motorB = 0, 0

        if robot_offset == 0:   # Detenerse
            speed_motorA = -1 * motorHandlerRaspberry.initial_speed
            speed_motorB = -1 * motorHandlerRaspberry.initial_speed
        elif robot_offset > safety_zone_range:  # Moverse a la izquierda
            speed_motorA = 10
            speed_motorB = -1 * motorHandlerRaspberry.initial_speed
        elif robot_offset < (- safety_zone_range):  # Moverse a la derecha
            speed_motorA = 10
            speed_motorB = -1 * motorHandlerRaspberry.initial_speed

        m1.set_motorA_speed(speed_motorA)
        m1.set_motorB_speed(speed_motorB)
        m1.forward()
