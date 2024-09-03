import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from werkzeug.serving import make_server
from waitress import serve
from waitress.server import create_server
import threading
from flask import Flask, request, jsonify
import numpy as np
import multiprocessing
from typing import List
import processing as ps

class MyFlaskApp:

    def __init__(self, window):
        self.app = Flask(__name__)
        self.window = window
        self.app.add_url_rule('/', 'home', self.home)

        self.server_started = False
        self.server = None
        self.thread = None

    def home(self):
        return "Hello, Flask!"

    def run_server(self,):
        self.server = create_server(self.app, host="localhost", port=5000)
        self.thread = threading.Thread(target=self.server.run)
        self.thread.start()
        self.server_started = True

    def close_server(self):
        if self.server_started:
            self.server.close()
            self.thread.join()
            self.server_started = False


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PyQt5 GUI App')
        self.setGeometry(100, 100, 400, 300)
        self.layout = QVBoxLayout()
        self.api_button = QPushButton('Start API', self)
        self.layout.addWidget(self.api_button)
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)
        # for flask api
        self.flask_app = MyFlaskApp(self)
        self.api_button.clicked.connect(self.toggle_api)


    def toggle_api(self):
        if self.flask_app.server_started:
            self.stop_api()
        else:
            self.start_api()

    def start_api(self):
        self.flask_app.run_server()
        self.api_button.setText('Stop API')

    def stop_api(self):

        if self.flask_app.server_started:
            self.flask_app.close_server()
            self.api_button.setText('Start API')
    
    def closeEvent(self, event):
        self.stop_api()
        event.accept()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Create an application object
    qt_app = QApplication(sys.argv)
    # Create a main window
    window = MainWindow()
    window.show()
    qt_app.exec()
