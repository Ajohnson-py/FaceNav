import threading
import time

from Quartz.CoreGraphics import (
    CGEventCreate, CGEventGetLocation, CGEventCreateMouseEvent,
    kCGEventMouseMoved, kCGEventLeftMouseDown, kCGEventLeftMouseUp,
    kCGEventRightMouseDown, kCGEventRightMouseUp, kCGMouseButtonLeft,
    CGEventPost, kCGHIDEventTap
)


class MouseHandler:
    def __init__(self, cursor_sensitivity):
        self.cursor_sensitivity = max(0.1, cursor_sensitivity)
        self.expression_action = None
        self.running = True

        # Start the listener in a separate thread
        self.listener_thread = threading.Thread(target=self._listen_for_expression, daemon=True)
        self.listener_thread.start()

    def _get_position(self) -> tuple[int, int]:
        """Private method that gets mouse position"""
        event = CGEventCreate(None)
        position = CGEventGetLocation(event)
        return position.x, position.y

    def _move_cursor(self, dx: int, dy: int) -> None:
        """Private method that moves the mouse relative to its current position"""
        x, y = self._get_position()
        step_x, step_y = int(dx * self.cursor_sensitivity), int(dy * self.cursor_sensitivity)

        num_steps = max(abs(dx), abs(dy))
        for _ in range(num_steps):
            x += step_x
            y += step_y

            event = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x, y), 0)
            CGEventPost(kCGHIDEventTap, event)

    def _click(self, button: str) -> None:
        """Private method that left or right clicks at current mouse position"""
        x, y = self._get_position()

        if button == "left":
            event_down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, event_down)

            event_up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, event_up)
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
