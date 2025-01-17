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

APP_VERSION = "1.3.4"

from PyQt5.QtWidgets import QApplication

from custom_widgets import MainWindow

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



 #%%   
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # create an application object
    app = QApplication(sys.argv)
    app.setApplicationVersion(APP_VERSION)
    # create a mainwindow
    window = MainWindow()
    window.show()
    window.show_about_dialog()
    window.activateWindow()

    app.exec_()


    # [k for k in dir(window) if k not in dir(QMainWindow) and not k.startswith('__')]
    # exit the application when the window is closed
    
    # sys.exit(app.exec_())    
