# -*- coding: utf-8 -*-
"""
Created on Sat May 20 09:11:45 2023

@author: Rakhul Raj
"""
import re
import os
import sys
from pathlib import Path
import time
import numpy as np
import requests
import jsonpickle
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5.uic import loadUi
import cv2
import processing as ps
from typing import Tuple, Union


from mymodule.utils import decorate_all_methods
from mymodule.exceptions import exception_handler


if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    application_path = Path(sys._MEIPASS)
else:
    application_path = Path(os.path.dirname(os.path.abspath(__file__)))


class Worker(QObject):
    state_updated = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, window: QMainWindow):
        super().__init__()
        self.window = window
        self._is_running = True

    def get_state(self):
        while self._is_running:
            if self.window.process_box.isChecked():
                url = "http://localhost:5454/api/current_state"
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        objects = response.json()
                        state = {
                            'measType': jsonpickle.decode(objects['measType']),
                            'binarize': jsonpickle.decode(objects['binarize'])
                        }

                        self.state_updated.emit(state)

                    else:
                        print("Failed to get edge image. Status code:", response.status_code)
                    time.sleep(0.2)

                except Exception as exp:
                    print(exp)
                    time.sleep(5)

            else:
                time.sleep(0.5)

    def stop(self):
        self._is_running = False
        self.finished.emit()



class ImageProcessor(QMainWindow):
    def __init__(self, worker: Worker = Worker):
        super().__init__()
        loadUi(application_path/ 'imageviewer.ui', self)  # Load the UI file
        self.current_folder_index = 0  # Initialize current_folder_index
        self.folders = []  # Initialize folders list
        self.images = []

        
        self.select_button.clicked.connect(self.select_folder)
        self.next_button.clicked.connect(self.next_folder)
        self.prev_button.clicked.connect(self.prev_folder)
        self.skip_forward_button.clicked.connect(self.skip_forward)
        self.skip_backward_button.clicked.connect(self.skip_backward)


        # for updating the state from modvortex
        self.prev_binarize = ps.Binarize_Type(ps.Binarize_Type.TYPE_CUSTOM)
        self.measType = None
        self.binarize = None

        self.worker = worker(window= self)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker.state_updated.connect(self.update_state)
        self.worker_thread.started.connect(self.worker.get_state)
        
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)        
        
        
        self.worker_thread.start()

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder:
            self.set_folder(folder= folder)
    
    def set_folder(self, folder : Union[str, Path]):

        # pattern for a number in the beginning
        pattern = re.compile(r'^[+-]?\d+(\.\d+)?')

        folder_path = Path(folder).parent

        self.folders = sorted([f for f in folder_path.iterdir() if f.is_dir() and pattern.match(f.name)], key=ps.sort_key)
        if self.folders:
            current_values = ps.sort_key(Path(folder))
            # loading the images from selected folder, in case if no folder is found the 1st folder is selected
            for ii, folder in enumerate(self.folders):
                if current_values == ps.sort_key(folder):
                    self.current_folder_index = ii
                    break
            else:
                self.current_folder_index = 0

            self.load_images()

    def sort_key(self, folder: Path): # going to be depreciated in the future versions
        
        name = folder.name
        parts = name.split('_')
        parts[0] = re.match(r'^[+-]?\d+(\.\d+)?', parts[0]).group()

        if parts[0] is None:
            raise ValueError('Folder Names Do_not match for folder {}'.format(str(folder)))
        
        if len(parts) == 2:
            try:
                return (float(parts[0][:-1]), int(parts[1]))
            except ValueError:
                pass
        
        return (float(parts[0][:-1]), 0)

    def load_images(self):
        if self.folders:
            folder = self.folders[self.current_folder_index]
            self.dir_label.setText(f'Current Directory: {folder}')
            if self.measType is None:
                img_type : Tuple = ('.png', '.jpeg', '.jpg', '.webp')
                image_files = [p for p in folder.iterdir() if np.any([p.name.lower().endswith(x) for x in img_type])]
                self.images = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)[:512]  for f in image_files]
            else:
                image_files = self.measType.settings.img_from_path(folder)
                self.images = [ps.load_image(f, self.measType) for f in image_files]
            images = self.images
            if images:
                if self.process_box.isChecked():
                    try:
                        # image = self.get_edge(images)
                        # image = self.get_edge_from_path(folder)
                        image = ps.get_edge(images, binarize= self.binarize, measType= self.measType)
                    except ConnectionError as ce:
                        return None
                else:
                    image =  images[2]
                # pixmap = QPixmap(str(images[0]))  # Load the first image for now
                qimage = self.qimage_fromdata(image)
                pixmap = QPixmap.fromImage(qimage)
                cropped_pixmap = pixmap.copy(0, 0, pixmap.width(), 512)  # Crop the height to 512 pixels
                self.label.setPixmap(cropped_pixmap)
                self.label.resize(cropped_pixmap.width(), cropped_pixmap.height())  # Resize QLabel to fit the cropped image

    def qimage_fromdata(self, img):
        '''define a function to convert NumPy array to QImage'''

        if len(img.shape) == 2:  # Grayscale image
            height, width = img.shape
            # qimg = QImage(img.data, width, height, QImage.Format_Grayscale8) # error was happening when i crop the width of the image
            qimg = QImage(bytes(img.data), width, height, QImage.Format_Grayscale8)
        elif len(img.shape) == 3:  # BGR image
            height, width, channels = img.shape
            if channels == 3:
                qimg = QImage(bytes(img.data), width, height, QImage.Format_BGR888)
            else:
                raise ValueError("Unsupported image format")
        else:
            raise ValueError("Unsupported image format")

        return qimg

    def next_folder(self):
        if self.current_folder_index < len(self.folders) - 1:
            self.current_folder_index += 1
            self.load_images()

    def prev_folder(self):
        if self.current_folder_index > 0:
            self.current_folder_index -= 1
            self.load_images()

    def skip_forward(self):
        current_value = ps.sort_key(self.folders[self.current_folder_index])

        for i in range(self.current_folder_index + 1, len(self.folders)):
            first, second = ps.sort_key(self.folders[i])
            if first > current_value[0] and second == current_value[1]:
                self.current_folder_index = i
                self.load_images()
                return
        else:

            if current_value[0] == ps.sort_key(self.folders[-1])[0]:
                for ii, folders in enumerate(self.folders):
                    first, second = ps.sort_key(folders)
                    # print('current value:', current_value, ' other:', (first, second ))
                    if current_value[1] == second:
                        self.current_folder_index = ii
                        self.load_images()
                        return

            else:
               self.current_folder_index -= 1
               self.skip_forward()

    def skip_backward(self):
        current_value = ps.sort_key(self.folders[self.current_folder_index])

        if self.current_folder_index != 0 :
            for i in range(self.current_folder_index - 1, -1, -1):
                first, second = ps.sort_key(self.folders[i])
                if first < current_value[0] and second == current_value[1]:
                    self.current_folder_index = i
                    self.load_images()
                    return
            else:
                self.current_folder_index -= 1
                self.skip_backward()
        else:
            self.current_folder_index = len(self.folders) -1
            self.load_images()


    def update_state(self, state):
        self.measType = state['measType']
        self.binarize = state['binarize']
        # print([not (x == y) for x,y in zip(self.binarize.__dict__.keys(),self.prev_binarize.__dict__.keys())])
        if np.any( [not (x == y) for x,y in zip(self.binarize.__dict__.values(),self.prev_binarize.__dict__.values())]):
            self.load_images()
            # print('updated_new')
        self.prev_binarize = self.binarize

    def get_edge_from_path(self, path:Path):

        'path is the folder containg the images'

        # URL of the Flask API endpoint
        url = "http://localhost:5454/api/edge_from_path"


        # Sample data to send to the endpoint
        data = str(path)

        # Send a POST request to the endpoint
        response = requests.post(url, json=data, headers={"Content-Type": "application/json"})

        # Check the response status code
        if response.status_code == 200:
            # Convert the response JSON content to a list of lists
            edge_image_list = response.json()

            # Convert the list of lists to a NumPy array
            edge_image_array = np.array(edge_image_list, dtype=np.uint8)
            return edge_image_array
            print("Edge Image Array:", edge_image_array)
        else:
            print("Failed to get edge image. Status code:", response.status_code)

    def get_edge(self, images):

        # URL of the Flask API endpoint
        url = "http://localhost:5454/api/edge"

        if images:
            # Sample data to send to the endpoint
            data = {f'image{ii}':x.tolist() for ii, x in enumerate(images)}

            # Send a POST request to the endpoint
            response = requests.post(url, json=data, headers={"Content-Type": "application/json"})

            # Check the response status code
            if response.status_code == 200:
                # Convert the response JSON content to a list of lists
                edge_image_list = response.json()

                # Convert the list of lists to a NumPy array
                edge_image_array = np.array(edge_image_list, dtype=np.uint8)
                return edge_image_array
                print("Edge Image Array:", edge_image_array)
            else:
                print("Failed to get edge image. Status code:", response.status_code)
    
    def closeEvent(self, event):
        self.worker.stop()
        self.worker_thread.quit()
        self.worker_thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = decorate_all_methods(exception_handler)(ImageProcessor)()
    # ex = ImageProcessor()
    ex.show()
    sys.exit(app.exec_())
