import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot

class Worker(QObject):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._is_running = True

    def run(self):
        while self._is_running:
            # Your long-running task here
            print("Worker is running...")
            QThread.sleep(1)

    def stop(self):
        self._is_running = False
        self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Window")
        self.setGeometry(100, 100, 600, 400)

        self.button = QPushButton("Open New Window", self)
        self.button.clicked.connect(self.open_new_window)
        self.setCentralWidget(self.button)

    def open_new_window(self):
        self.new_window = NewWindow(self)
        self.new_window.show()

class NewWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("New Window")
        self.setGeometry(200, 200, 400, 300)

        self.thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.thread.start()

    def closeEvent(self, event):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
