import threading
import time

from Quartz.CoreGraphics import (
    CGEventCreate, CGEventGetLocation, CGEventCreateMouseEvent,
    kCGEventMouseMoved, kCGEventLeftMouseDown, kCGEventLeftMouseUp,
    kCGEventRightMouseDown, kCGEventRightMouseUp, kCGMouseButtonLeft,
    CGEventPost, kCGHIDEventTap, CGDisplayPixelsWide, CGDisplayPixelsHigh,
    CGMainDisplayID
)


class MouseHandler:
    def __init__(self, cursor_sensitivity):
        self.cursor_sensitivity = max(0.1, cursor_sensitivity)
        self.expression_action = None
        self.running = True

        self.speed_multiplier = 1.0
        self.speed_increment = 0.1
        self.max_speed = 6
        self.last_move_time = time.time()

        # Start the listener in a separate thread
        self.listener_thread = threading.Thread(target=self._listen_for_expression, daemon=True)
        self.listener_thread.start()

    def _get_position(self) -> tuple[int, int]:
        """Private method that gets mouse position"""
        event = CGEventCreate(None)
        position = CGEventGetLocation(event)
        return position.x, position.y

    def _move_cursor(self, dx: int, dy: int) -> None:
        """Private method that moves the cursor relative to its current position at increasing speed"""
        # Reset speed multiplier if no movement
        if dx == 0 and dy == 0:
            self.speed_multiplier = 1.0
            return

        # Increase speed over time, but cap it at max_speed
        if time.time() - self.last_move_time < 0.25:  # If moving continuously
            self.speed_multiplier = min(self.speed_multiplier + self.speed_increment, self.max_speed)
        else:
            self.speed_multiplier = 1.0

        self.last_move_time = time.time()

        x, y = self._get_position()
        screen_width = CGDisplayPixelsWide(CGMainDisplayID())
        screen_height = CGDisplayPixelsHigh(CGMainDisplayID())

        step_x, step_y = int(dx * self.cursor_sensitivity * self.speed_multiplier), \
            int(dy * self.cursor_sensitivity * self.speed_multiplier)

        # Move cursor one pixel at a time in dx and dy
        num_steps = max(abs(dx), abs(dy))
        for _ in range(num_steps):
            # Clamp cursor position to stay within the screen
            x = max(0, min(screen_width - 1, x + step_x))
            y = max(0, min(screen_height - 1, y + step_y))

            event = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x, y), 0)
            CGEventPost(kCGHIDEventTap, event)
            time.sleep(0.01)

    def _click(self, button: str) -> None:
        """Private method that left or right clicks at current mouse position"""
        x, y = self._get_position()

        # Left click
        if button == "left":
            event_down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, event_down)

            event_up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, event_up)
        # Right click
        elif button == "right":
            event_down = CGEventCreateMouseEvent(None, kCGEventRightMouseDown, (x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, event_down)

            event_up = CGEventCreateMouseEvent(None, kCGEventRightMouseUp, (x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, event_up)

    def _listen_for_expression(self):
        """Background thread: listens for facial expression actions"""
        while self.running:
            if self.expression_action == "clickLeft":
                self._click("left")
                self.expression_action = None
            elif self.expression_action == "clickRight":
                self._click("right")
                self.expression_action = None
            elif isinstance(self.expression_action, tuple):
                dx, dy = self.expression_action
                self._move_cursor(dx, dy)
                self.expression_action = None

            time.sleep(0.01)
