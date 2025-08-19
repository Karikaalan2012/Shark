import RPi.GPIO as GPIO
from time import sleep

sevo_pin = 14
delay = 0.5

GPIO.setwarnings(False)

GPIO.setmode(GPIO.BOARD)

GPIO.setup(servo_pin, GPIO.OUT)

pwm = GPIO.PWM(servo_pin, 50)
pwm.start(0)

pwm.ChangeDutyCycle(5)
sleep(delay)
pwm.ChangeDutyCycle(7.5)
sleep(delay)
pwm.ChangeDutyCycle(10)
sleep(delay)

pwm.stop(0)

GPIO.cleanup()
