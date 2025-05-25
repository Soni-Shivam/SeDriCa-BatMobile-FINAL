from pynput import keyboard
from gpiozero import AngularServo, Motor

left_motor = Motor(forward=13, backward=19, pwm=True)
right_motor = Motor(forward=12, backward=18, pwm=True)


movement_state = None


def control_motors(speed_L, speed_R):
    # Convert 0-100 to -1 to 1 range
    def to_motor_val(speed, direction):
        val = speed / 100.0
        return val if direction else -val
    # speed_L and speed_R are tuples of the form (speed : 0-100, direction:true/false)

    left_motor.value = to_motor_val(speed_L[0], speed_L[1])
    right_motor.value = to_motor_val(speed_R[0], speed_R[1])



def update_movement(new_state):
    global movement_state
    if new_state != movement_state:
        print(f"{'Stopping' if new_state is None else 'Moving ' + new_state}")
        movement_state = new_state

def on_press(key):
    try:
        if key == keyboard.Key.up:
            update_movement("forward")
            control_motors((100, True), (100, True))
        elif key == keyboard.Key.down:
            update_movement("backward")
            control_motors((100, False), (100, False))
        elif key == keyboard.Key.left:
            update_movement("left")
            control_motors((100, False), (100, True))  
        elif key == keyboard.Key.right:
            update_movement("right")
            control_motors((100, True), (100, False))
    except Exception as e:
        print(f"Error: {e}")

def on_release(key):
    if key in {keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right}:
        update_movement(None)
        control_motors((0, True), (0, False))

    elif key == keyboard.Key.esc:
        print("Exiting...")
        return False  # Stop the listener

print("Car controller started. Use arrow keys to control the car. ESC to exit.")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
