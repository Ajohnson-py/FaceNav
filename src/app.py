from Cocoa import NSApplication, NSApp, NSStatusBar, NSVariableStatusItemLength, NSMenu, NSMenuItem, NSObject
from PyObjCTools import AppHelper


class AppDelegate(NSObject):
    def __init__(self):
        self.not_paused = None

    def setNotPaused_(self, status: bool):
        self.not_paused = status

    def applicationDidFinishLaunching_(self, notification):
        # Set activation policy so that no dock app appears
        NSApp.setActivationPolicy_(2)

        # Create menu bar item
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self.status_item.setTitle_("FaceNav")

        # Create menu for menu bar item
        menu = NSMenu.alloc().init()

        pause_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Pause", "pause:", "")
        pause_item.setTarget_(self)
        menu.addItem_(pause_item)

        # Create Quit item
        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Quit", "terminate:", "q")
        quit_item.setTarget_(NSApp)
        menu.addItem_(quit_item)

        # Apply menu to menu bar item
        self.status_item.setMenu_(menu)

    def pause_(self, sender):
        self.not_paused.value = not self.not_paused.value


def start_menu_bar_app(paused: bool):
    app = NSApplication.sharedApplication()
    app_delegate = AppDelegate()
    app_delegate.setNotPaused_(paused)
    NSApp.setDelegate_(app_delegate)

    # Run the event loop
    AppHelper.runEventLoop()