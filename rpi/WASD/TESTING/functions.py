from gpiozero import AngularServo, Motor

left_motor = Motor(forward=12, backward=13, pwm=True)
right_motor = Motor(forward=22, backward=23, pwm=True)


def control_motors(speed_L, speed_R):
    # Convert 0-100 to -1 to 1 range
    def to_motor_val(speed, direction):
        val = speed / 100.0
        return val if direction else -val
    # speed_L and speed_R are tuples of the form (speed : 0-100, direction:true/false)

    left_motor.value = to_motor_val(speed_L[0], speed_L[1])
    right_motor.value = to_motor_val(speed_R[0], speed_R[1])
        
if w:
    control_motors((100, True), (100, True))
elif s:
    control_motors((100, False), (100, False))
elif a:
    control_motors((100, False), (100, True))  
elif d:
    control_motors((100, True), (100, False))
elif q:
    quit()