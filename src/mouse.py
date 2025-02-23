from Quartz.CoreGraphics import (
    CGEventCreate, CGEventGetLocation, CGEventCreateMouseEvent,
    kCGEventMouseMoved, kCGEventLeftMouseDown, kCGEventLeftMouseUp,
    kCGEventLeftMouseDragged, kCGMouseButtonLeft, CGEventPost, kCGHIDEventTap
)
import time


class MouseHandler:
    def __init__(self, cursor_sensitivity ):
        self.cursor_sensitivity = max(0.1, cursor_sensitivity)

    def _getPosition(self) -> tuple[int, int]:
        event = CGEventCreate(None)
        position = CGEventGetLocation(event)
        return position.x, position.y

    def move_cursor(self, dx, dy):
        # TODO: Add better mouse sensitivity handling
        x, y = self._getPosition()
        step_x, step_y = int(dx * self.cursor_sensitivity), int(dy * self.cursor_sensitivity)

        num_steps = max(abs(dx), abs(dy))
        for _ in range(num_steps):
            x += step_x
            y += step_y

            event = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x, y), 0)
            CGEventPost(kCGHIDEventTap, event)

            #time.sleep(self.cursor_sensitivity)
