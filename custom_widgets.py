# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:01:56 2023

@author: Rakhul Raj
"""

from PyQt5.QtWidgets import (QPushButton, QLabel)
from PyQt5.QtGui import QMouseEvent, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
import numpy as np



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
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.first_click = None
        self.second_click = None
        self.last_pos = None
        self.outline = 0
        
        
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



        
        
        