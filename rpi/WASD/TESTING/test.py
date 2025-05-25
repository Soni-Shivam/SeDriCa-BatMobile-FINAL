import curses
from gpiozero import Motor
import time

# Motor setup
left_motor = Motor(forward=17, backward=12, pwm=True)
right_motor = Motor(forward=27, backward=18, pwm=True)

# Global movement state
movement_state = None

def control_motors(speed_L, speed_R):
    # Convert 0-100 to -1 to 1 range
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
        stdscr.addstr(2, 0, msg.ljust(40))  # Overwrites old message
        stdscr.refresh()

def main(stdscr):
    # Initialize curses settings
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    stdscr.clear()
    stdscr.addstr(0, 0, "Use arrow keys to control the car. Press Q to exit.")
    stdscr.refresh()

    while True:
        key = stdscr.getch()
        if key == -1:
            continue

        if key == curses.KEY_UP:
            update_movement("forward", stdscr)
            control_motors((100, True), (100, True))
        elif key == curses.KEY_DOWN:
            update_movement("backward", stdscr)
            control_motors((100, False), (100, False))
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

        time.sleep(0.05)

if __name__ == "__main__":
    curses.wrapper(main)
