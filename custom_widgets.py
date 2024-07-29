# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:01:56 2023

@author: Rakhul Raj
"""

import sys
import os
from pathlib import Path
from PyQt5 import uic
from PyQt5.QtWidgets import (QWidget, QFileDialog, QPushButton, QLabel, QMainWindow, QDialog)
from PyQt5.QtGui import QMouseEvent, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
import numpy as np
import pandas as pd

if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    application_path = Path(sys._MEIPASS)
else:
    application_path = Path(os.path.dirname(os.path.abspath(__file__)))

class ActionButton(QPushButton):
    '''An extension of a QPushButton that supports QAction.
    This class represents a QPushButton extension that can be
    connected to an action and that configures itself depending
    on the status of the action.
    When the action changes its state, the button reflects
    such changes, and when the button is clicked the action
    is triggered.'''
    # The action associated to this button.
    actionOwner = None

    # Parent the widget parent of this button
    def __init__(self, parent=None):
        super().__init__(parent)

    # Set the action owner of this button, that is the action
    # associated to the button. The button is configured immediately
    # depending on the action status and the button and the action
    # are connected together so that when the action is changed the button
    # is updated and when the button is clicked the action is triggered.
    # action the action to associate to this button
    def setAction(self, action):
        # if I've got already an action associated to the button
        # remove all 
        for action in self.actions():
            self.removeAction(action)
        self.addAction(action)
        if self.actionOwner and self.actionOwner != action:
            self.actionOwner.changed.disconnect(self.updateButtonStatusFromAction)
            self.clicked.disconnect(self.actionOwner.trigger)

        # store the action
        self.actionOwner = action

        # configure the button
        self.updateButtonStatusFromAction()

        # connect the action and the button
        # so that when the action is changed the
        # button is changed too!
        self.actionOwner.changed.connect(self.updateButtonStatusFromAction)
        self.clicked.connect(self.actionOwner.trigger)

    # Update the button status depending on a change
    # on the action status. This slot is invoked each time the action
    # "changed" signal is emitted.
    def updateButtonStatusFromAction(self):
        if not self.actionOwner:
            return
        # self.setText(self.actionOwner.text())
        self.setStatusTip(self.actionOwner.statusTip())
        self.setToolTip(self.actionOwner.toolTip())
        self.setIcon(self.actionOwner.icon())
        self.setEnabled(self.actionOwner.isEnabled())
        self.setCheckable(self.actionOwner.isCheckable())
        self.setChecked(self.actionOwner.isChecked())


class MyLabel(QLabel):
    
    def __init__(self, mainwindow: QMainWindow = None, parent=None):
        super().__init__(parent)
        self.mainwindow = mainwindow
        self.setMouseTracking(True)
        self.first_click: QPoint = None
        self.second_click: QPoint = None
        self.last_pos: QPoint = None
        self.outline: int = 0
        
        
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # If the left button is pressed, update the position of the second click and redraw the widget.
        if event.buttons() == Qt.LeftButton:
            self.second_click = event.pos()
            self.update()
        # If the right button is pressed, update the position of the last click.
        elif event.buttons() == Qt.RightButton:
            if self.last_pos is None:
                self.last_pos = event.pos()
            else:
                # Calculate the difference between the current and last position and update the first and second click positions.
                position = event.pos() - self.last_pos
                self.last_pos = event.pos()
                if self.first_click is not None and self.second_click is not None:
                    self.first_click += position
                    self.second_click += position
                    self.update()
        
        # for updating the x, y position in another lablel
        x: int = event.x()  # Get x-coordinate of the mouse
        y: int = event.y()  # Get y-coordinate of the mouse
        self.mainwindow.update_position_label(str(x), str(y)) # Update the parent widget with the coordinates
    
    def leaveEvent(self, event: QMouseEvent) -> None:
        self.mainwindow.update_position_label('...', '...')
                
    def mouseReleaseEvent(self, event):
        # If the right button is released, set the last position to None.
        if event.button() == Qt.RightButton:  
            self.last_pos = None
            
    def mousePressEvent(self, event):
        # If the left button is pressed, set the position of the first or second click.
        if event.button() == Qt.LeftButton:
            if self.first_click is None:
                self.first_click = event.pos()
            elif self.second_click is None:
                self.second_click = event.pos()
                self.update()
            else:
                self.second_click = event.pos()
                self.update()
        # If both left and right buttons are pressed, clear the line
        elif event.buttons() == Qt.LeftButton | Qt.RightButton:
            self.clear()
                
    def paintEvent(self, event):
        super().paintEvent(event)
        # If both first and second click positions are not None, draw a red line between them.
        if self.first_click is not None and self.second_click is not None:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2)
            painter.setPen(pen)
            painter.drawLine(self.first_click, self.second_click)
            
            if self.outline > 0:
    
                pen = QPen(Qt.green, 2)
                painter.setPen(pen)
                x2 = self.first_click.x()
                y2 = self.first_click.y()
                x1 = self.second_click.x()
                y1 = self.second_click.y()
                
                if (x2 - x1) == 0:
                  
                    xx = self.outline
                    yy = 0 
                
                else:
                    
                    m = (y2-y1) / (x2-x1)
                    xx = int(np.round(m/np.sqrt(1+m**2) * self.outline))
                    yy = int(np.round(-1/np.sqrt(1+m**2) * self.outline))
                
                new_first_click = QPoint( x2+xx, y2+yy )
                new_second_click =  QPoint( x1+xx, y1+yy )
                painter.drawLine(new_first_click, new_second_click)
                
                xx = -xx
                yy = -yy
                
                new_first_click = QPoint( x2+xx, y2+yy )
                new_second_click =  QPoint( x1+xx, y1+yy )
                painter.drawLine(new_first_click, new_second_click)
            
    def set_line_points(self, p1, p2):
        
        self.first_click = p1
        self.second_click = p2
        self.update()
        
    def set_outline(self, outline):
        
        self.outline = outline
        self.update()

    def clear(self):

        self.first_click = None
        self.second_click = None
        self.update()
        
    # def setPixmap()


# class SettingWindow(QWidgets):

#     def __init__(self, parent = None):
        
#         super().__init__(parent)
#         uic.loadUi(application_path/"trialwindow.ui", self)

#         self.scale: float
#         self.kernel: int
#         self.img_width: int
#         self.img_height: int

#         self.set_constants(application_path/"settings"/"default.set")
    
#     def set_constants(self, path):

#         df = pd.read_csv(path, sep = '=', header=None, index_col= 0, skip_blank_lines= True, comment='#')
#         data = df.iloc[0].to_dict()

#         key_list = ['scale', 'kernel', 'img_width', 'img_height' ]

#         if set(key_list) == set(data.keys()):
#             self.__dict__.update(data)
#         else:
#             raise ValueError('Invalid Settings file')
    
#     def set_value(self):

#         self.scale = self.scale_box.value()
#         self.kernel = self.gaussian_box.value()
#         self.img_width = self.img_width_box.value()
#         self.img_height = self.img_height_box.value()

class SettingWindow(QDialog):
    

    def __init__(self, parent: QWidget = None) -> None:
        """
        Initialize the SettingWindow class.

        :param parent: The parent widget, if any.
        """
        super().__init__(parent)
        uic.loadUi(application_path / "settings.ui", self)

        # # Set the window flag to keep the window on top of its parent
        # self.setWindowFlags(self.windowFlags() | Qt.SubWindow)
        # self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        self.setWindowModality(Qt.WindowModal) # it willnot allow input in the parent window

        # Instance variables for settings
        self.scale: float
        self.kernel: int
        self.img_width: int
        self.img_height: int

        self.key_list = ['scale', 'kernel', 'img_width', 'img_height'] # these are the attributes of this subclass

        # Load default settings
        self.set_constants(application_path / "settings" / "default.set")

        # Connect buttons to their respective functions
        self.load_button.clicked.connect(self.load_preset)
        self.save_button.clicked.connect(self.save_preset)
        self.set_button.clicked.connect(self.set_value)

        # Connect value change signals to the check_changes function
        self.scale_box.valueChanged.connect(self.check_changes)
        self.gaussian_box.valueChanged.connect(self.check_changes)
        self.img_width_box.valueChanged.connect(self.check_changes)
        self.img_height_box.valueChanged.connect(self.check_changes)

        # Initial check to update the set button state
        self.check_changes()

    def set_constants(self, path: Path) -> None:
        """
        Set the constants from a settings file.

        :param path: Path to the settings file.
        :raises ValueError: If the settings file is invalid.
        """
        df = pd.read_csv(path, sep='=', header=None, index_col=0, skip_blank_lines=True, comment='#').T
        df.columns = map(lambda x: x.strip(), df.columns)
        data = df.iloc[0].to_dict()

        key_list = self.key_list
        # Check if the settings file contains all required keys
        if set(key_list) == set(data.keys()):
            data['scale'] = float(data['scale'])
            data['kernel'] = int(data['kernel'])
            data['img_width'] = int(data['img_width'])
            data['img_height'] = int(data['img_height'])
            self.__dict__.update(data)
        else:
            raise ValueError('Invalid Settings file')

        # Update the widget values based on the loaded settings
        self.update_widgets()

    def update_widgets(self) -> None:
        """
        Update the widget values based on the instance variables.
        """
        self.scale_box.setValue(self.scale)
        self.gaussian_box.setValue(self.kernel)
        self.img_width_box.setValue(self.img_width)
        self.img_height_box.setValue(self.img_height)

    def set_value(self) -> None:
        """
        Set the instance variables from the widget values.
        """
        self.scale = self.scale_box.value()
        self.kernel = self.gaussian_box.value()
        self.img_width = self.img_width_box.value()
        self.img_height = self.img_height_box.value()
        self.check_changes()

    def load_preset(self) -> None:
        """
        Open a file dialog to select a settings file and update the widget values.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Preset",
                                                    str(application_path / "settings"), 
                                                    "Settings Files (*.set);;All Files (*)", 
                                                    options=options)
        if file_path:
            df = pd.read_csv(Path(file_path), sep='=', header=None, index_col=0, skip_blank_lines=True, comment='#').T
            df.columns = map(lambda x: x.strip(), df.columns)
            data = df.iloc[0].to_dict()

            key_list = self.key_list

            # Check if the settings file contains all required keys
            if set(key_list) == set(data.keys()):
                self.scale_box.setValue(float(data['scale']))
                self.gaussian_box.setValue(int(data['kernel']))
                self.img_width_box.setValue(int(data['img_width']))
                self.img_height_box.setValue(int(data['img_height']))
            else:
                raise ValueError('Invalid Settings file')
            
    def save_preset(self) -> None:
        """
        Open a file dialog to select a save directory and file name, then save the current settings.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Preset",
                                                    str(application_path / "settings"),
                                                    "Settings Files (*.set);;All Files (*)",
                                                    options=options)
        if file_path:
            data = {
                'scale': self.scale_box.value(),
                'kernel': self.gaussian_box.value(),
                'img_width': self.img_width_box.value(),
                'img_height': self.img_height_box.value()
            }
            df = pd.DataFrame(list(data.items()), columns=['Parameter', 'Value'], dtype= 'object')
            df.to_csv(file_path, sep='=', header=False, index=False)

    def check_changes(self) -> None:
        """
        Enable or disable the set button based on whether the widget values differ from the instance variables.
        """
        if (self.scale_box.value() != self.scale or
            self.gaussian_box.value() != self.kernel or
            self.img_width_box.value() != self.img_width or
            self.img_height_box.value() != self.img_height):
            self.set_button.setEnabled(True)
        else:
            self.set_button.setEnabled(False)

    def show(self) -> None:
        """
        Override the show method to update widgets only if the window was previously closed.
        """
        if not self.isVisible():
            self.update_widgets()
        super().show()

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    wind = SettingWindow()
    wind.show()
    app.exec()