import curses
from gpiozero import Motor
import time

# Motor setup
left_motor = Motor(forward=17, backward=27, pwm=True)
right_motor = Motor(forward=18, backward=12, pwm=True)
power_constant =100
turn_const =55
# Global movement state
movement_state = None

def control_motors(speed_L, speed_R):
    def to_motor_val(speed, direction):
        val = speed / 100.0
        return val if direction else -val

    left_motor.value = to_motor_val(speed_L[0], speed_L[1])
    right_motor.value = to_motor_val(speed_R[0], speed_R[1])

def update_movement(new_state, stdscr):
    global movement_state
    if new_state != movement_state:
        movement_state = new_state
        msg = 'Stopping' if new_state is None else 'Moving ' + new_state
        stdscr.addstr(2, 0, msg.ljust(40))
        stdscr.refresh()

def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    stdscr.clear()
    stdscr.addstr(0, 0, "Use arrow keys to control the car. Press Q to exit.")
    stdscr.refresh()

    last_key_time = time.time()
    idle_timeout = 0.1  # seconds

    while True:
        key = stdscr.getch()
        current_time = time.time()

        if key == -1:
            # Check for idle timeout
            if movement_state is not None and (current_time - last_key_time > idle_timeout):
                update_movement(None, stdscr)
                control_motors((0, True), (0, False))
            # time.sleep(0.05)
            continue

        last_key_time = current_time  # Update last key press time

        if key == curses.KEY_UP:
            update_movement("forward", stdscr)
            control_motors((power_constant, True), (power_constant, True))
        elif key == curses.KEY_DOWN:
            update_movement("backward", stdscr)
            control_motors((power_constant, False), (power_constant, False))
        elif key == curses.KEY_LEFT:
            update_movement("left", stdscr)
            control_motors((100, False), (100, True))
        elif key == curses.KEY_RIGHT:
            update_movement("right", stdscr)
            control_motors((100, True), (100, False))
        elif key in [ord('q'), ord('Q')]:
            update_movement(None, stdscr)
            control_motors((0, True), (0, False))
            break
        else:
            update_movement(None, stdscr)
            control_motors((0, True), (0, False))


if __name__ == "__main__":
    curses.wrapper(main)
