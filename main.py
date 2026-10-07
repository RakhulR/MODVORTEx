# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:24:16 2023

@author: Rakhul Raj
"""
#%%
import sys
import os
import multiprocessing
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

APP_VERSION = "1.4.0"

try:
    # This will only work in IPython
    ipython = get_ipython()
    # Use the magic command here
    ipython.run_line_magic("matplotlib", '')
except NameError:
    pass

if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    application_path = Path(sys._MEIPASS)
else:
    application_path = Path(os.path.dirname(os.path.abspath(__file__)))

class ModvortexLoadingWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        version = QApplication.instance().applicationVersion()
        
        # Keep the frameless/stays-on-top flags
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Main layout for the widget
        self.main_layout = QVBoxLayout(self)

        # --- THE FIX: A Container Frame ---
        # We put everything inside a frame so we can give it a background color
        self.container = QFrame()
        self.container.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 30, 220); /* Dark grey, slightly see-through */
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 30);
            }
        """)
        
        # Layout for inside the container
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(20, 20, 20, 20)

        # 1. Logo
        self.logo = QLabel()
        pix = QPixmap("icons/loading_circles_blue_gradient.png").scaledToWidth(500, Qt.SmoothTransformation)
        self.logo.setPixmap(pix)
        self.logo.setAlignment(Qt.AlignCenter)
        self.container_layout.addWidget(self.logo)

        # 2. Status Label
        self.status_label = QLabel("MODVORTEx is loading, please wait...")
        self.status_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold; background: transparent;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.container_layout.addWidget(self.status_label)

        # 3. Version Label
        self.version_label = QLabel(f"Version {version}")
        self.version_label.setStyleSheet("color: #888888; font-size: 12px; background: transparent;")
        self.version_label.setAlignment(Qt.AlignRight)
        self.container_layout.addWidget(self.version_label)

        # Add the container to the main widget layout
        self.main_layout.addWidget(self.container)

        # --- THE FIX: Force window to fit content ---
        self.adjustSize() 
        
        # Center the window on the screen
        self.center_on_screen()

    def center_on_screen(self):
        qr = self.frameGeometry()
        cp = QApplication.desktop().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def mousePressEvent(self, event):
        pass

class HeavyLoader(QThread):
    finished_signal = pyqtSignal(object)

    def run(self):
        # All the "slow" imports and AI initialization happen here
        from custom_widgets import MainWindow
        self.finished_signal.emit(MainWindow)

 #%%   
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # create an application object
    app = QApplication(sys.argv)
    app.setApplicationVersion(APP_VERSION)
    loader_gui = None
    def launch_main_app(MainWindowClass):
        global window
        # Initialize the main window class
        window = MainWindowClass()
        
        # Transfer control: Close loader, show main app
        loader_gui.close()
        window.show()
        window.raise_()
        window.activateWindow()
        
        if hasattr(window, 'show_about_dialog'):
            window.show_about_dialog()

    # Start the loading thread
    thread = HeavyLoader()
    thread.finished_signal.connect(launch_main_app)
    thread.start()
    
    # Launch the branded loader with the version number
    loader_gui = ModvortexLoadingWindow()
    loader_gui.show()

    app.exec_()


    # [k for k in dir(window) if k not in dir(QMainWindow) and not k.startswith('__')]
    # exit the application when the window is closed
    
    # sys.exit(app.exec_())    
