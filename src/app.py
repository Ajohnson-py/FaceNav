import subprocess
import rumps


class MenuBarApp(rumps.App):
    def __init__(self, running):
        """Menu bar app runs in a separate process and controls `running` flag."""

        def is_dark_mode() -> bool:
            """Returns True if macOS is in dark mode, even in Auto mode"""
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True,
                text=True
            )
            return result.stdout.strip() == "Dark"

        super(MenuBarApp, self).__init__("FaceNav")

        if is_dark_mode():
            self.icon = "./Resources/happy-mac-dark.png"
        else:
            self.icon = "./Resources/happy-mac-light.png"

        self.menu = ["Pause"]
        self.running = running

    @rumps.clicked("Pause")
    def pause_app(self, _):
        """Toggles face detection on/off."""
        self.running.value = not self.running.value
        self.menu = ["Resume"]
