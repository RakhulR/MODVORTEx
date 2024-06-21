# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:24:16 2023

@author: Rakhul Raj
"""
import sys
import re
import os
import pathlib
from pathlib import Path
import pandas as pd
import cv2
from PyQt5.QtWidgets import (QMainWindow,
                             QApplication,
                             QMessageBox,
                             QFileDialog,
                             QLabel, 
                             QShortcut,
                             QSizePolicy,
                             QWidget
    )
from PyQt5 import uic
from PyQt5.QtCore import QSize, QPoint, Qt
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
import numpy as np
import icons
from custom_widgets import MyLabel
import processing as ps
# iconSize = QSize(3000,3000)
import multiprocessing
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
from mymodule.utils import decorate_all_methods
from mymodule.exceptions import exception_handler

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


def do(volt, paths, out_path, measurmentType, binarize):
    
    out = ps.float_str(ps.field(float(volt)/10),2) + 'mT.txt'
    
    dat = ps.dt_curve(paths, volt, measurmentType, binarize)
    
    mt = measurmentType
    bi = binarize
    
    string = f'# point2 : {mt.point2},  point1 : {mt.point1},  DCenter : {mt.center},  Binarize : {bi.threshold},\
  outline : {mt.outline}\n'
        
    with open(out_path/out, 'w') as f:
        f.write(string)
    dat.to_csv(out_path/out, mode = 'a',  sep = '\t', index = False)

@decorate_all_methods(exception_handler)
class MainWindow(QMainWindow):
    
    def __init__(self):
        
        super(MainWindow, self).__init__()
        uic.loadUi(application_path/"mainwindow_v.ui", self)
        # self.toolButton.setIconSize(iconSize)
        # setting slot for the qaction
        self.actionBinarizeImage.triggered.connect(self.display_binarized)
        # connecting button with qaction
        self.binarize_button.setAction(self.actionBinarizeImage)
        
        # changing the binarze combobox and spinbox rebinarize the image with new values
        self.b_combo_box.currentIndexChanged.connect(lambda : self.display_binarized(True) 
                                             if self.binarize_button.isChecked()
                                             else self.display_binarized(False))
        self.spinBox.valueChanged.connect(lambda : self.display_binarized(True) 
                                             if self.binarize_button.isChecked()
                                             else self.display_binarized(False))
        # disable the threshold value selection for otsu algorithm
        self.b_combo_box.currentIndexChanged.connect(lambda index : 
                                                     self.spinBox.setEnabled(not index)
                                                     )
            
        self.inverse.stateChanged.connect(lambda : self.display_binarized(True) if self.binarize_button.isChecked()
                                          else None)
            
        self.setButton.clicked.connect(self.set_direction)
        self.loadPointsButton.clicked.connect(self.def_direction)
        self.dial.valueChanged.connect(self.outline_dial)
        self.genEdges.clicked.connect(self.add_edge)
        self.calcButton.clicked.connect(self.calculate_motion)
        self.calcAllButton.clicked.connect(self.calculate_all)
        self.bulkcalcButton.clicked.connect(self.calculate_all_from_parent)
        
            
        self.loadfolder.clicked.connect(self.loadfolder_f)
        self.plot_button.clicked.connect(lambda  : self.plot(Path(self.textInputFolder.toPlainText()).parent/
                                                             self.save_folder.text()                                                             
                                                             ))
        
        
        # Test
        # self.load_options.clicked.connect(lambda : print(type(self.tabWidget.widget(0))))
        # self.load_options.clicked.connect(lambda : print(self.tabWidget.widget(0).size()))
        # self.load_options.clicked.connect(self.def_direction)
        
        # load the images while clicking the load button
        # self.textInputFolder.textChanged.connect(self.load_images)
        self.load.clicked.connect(lambda : self.load_images()) # lambda is used bcoz signal gives\
                                                        # something to the slot which is undesirable
        
        # when tab clicked tab closed button
        self.tabWidget.tabCloseRequested.connect(self.on_tab_close_requested)
        
        # shortcut for moving left and right across the tabs
        self.tabWidget.shortcut_left = QShortcut(QKeySequence("Left"), self)
        self.tabWidget.shortcut_right = QShortcut(QKeySequence("Right"), self)
        self.tabWidget.shortcut_left.activated.connect(lambda : self.move_tab('left'))
        self.tabWidget.shortcut_right.activated.connect(lambda : self.move_tab('right'))
        
        self.tabWidget.currentChanged.connect(lambda : self.update_th_label())
        # self.show()
        
        # images that are loaded in the tabs
        self.images = None
        # save the threshold value while binarizing
        self.threshold = None
        
        # set the file filter to show only .py files
        # dialog.setNameFilter("Python files (*.png)")
    
    def loadfolder_f(self):
        # create a file dialog object
        dialog = QFileDialog()
        # dialog.setFileMode(QFileDialog.AnyFile)
        # # set the option to show files and directories
        # dialog.setOption(QFileDialog.DontUseNativeDialog, True)

        # folder = dialog.getExistingDirectory(self, 'rat','' , QFileDialog.DontUseNativeDialog)
        # folder.setOption(QFileDialog.ShowDirsOnly)
        
        folder = dialog.getOpenFileName(self, "Select Directory", "", "Directory (*)")
        if folder[0]:
        #     # adding the file to the text box.
            res = pathlib.Path(folder[0]).parent
            self.textInputFolder.setPlainText(str(res))

    def load_images(self, update_images = None):
        '''
        load the images to the tabs form the inputfolder if no update_images are given.
        or update the images if update_images are given

        Parameters
        ----------
        update_images : list, optional
            Image to update if needed. The default is None.

        Returns
        -------
        None.

        '''
        # extracting the text from the textbox
        # self.tabWidget.clear()


            
        if update_images == None:
            
            # clearing all tabs
            while self.tabWidget.count():
                self.tabWidget.widget(0).deleteLater()
                self.tabWidget.removeTab(0)
                
                
            text = self.textInputFolder.toPlainText()
            path = pathlib.Path(text)
            images =[* path.glob('*.png')][::1]
        
            images = [cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)[:512] for image in images]
            self.images = images
            images = [self.qimage_fromdata(image) for image in images]
            
            tabs = [MyLabel() for _ in images]
            [label.resize(image.size().width(), image.size().height()) for label, image in zip(tabs,images)]
            [label.setMaximumSize(image.size().width(), image.size().height()) for label, image in zip(tabs,images)]
            [tab.setPixmap(QPixmap.fromImage(image))for tab, image in zip(tabs,images)]
            
            for ii, tab in enumerate(tabs):
                
                self.tabWidget.addTab(tab, f"tab{ii}")
                

        else :
            
            images = [self.qimage_fromdata(image) for image in update_images]
            
            tabs = [self.tabWidget.widget(i) for i in range(self.tabWidget.count()) if self.tabWidget.widget(i)]
            
            [tab.setPixmap(QPixmap.fromImage(image))for tab, image in zip(tabs,images)]
            

        
    def update_th_label(self):
        "function to set the threshold label"
        if self.binarize_button.isChecked():
            ind = self.tabWidget.currentIndex()
            
            if len(self.threshold) > ind:
                
                self.label_3.setText(f"Current Threshold Value    : {self.threshold[ind]}")
                
            else: #ind is equal to len(threshold) means that the last tab is the edge tab
                
                self.label_3.setText("Current Threshold Value    : None")
        else:
            
            self.threshold = None
            self.label_3.setText("Current Threshold Value    : None")
            
            
            
    
    def display_binarized(self, satus):
        
        current_index = self.tabWidget.currentIndex()
        
        if satus:
            
            # data_img  = [self.qimage_to_numpy(image) for image in self.images]
            data_img  = self.images
            bin_type = ps.Binarize_Type.from_window(self)
            if  bin_type == 1:
                
                # binarized_img = [self.otsu_binarize(image) for image in data_img]
                # this loop below merges the first images with the last image horizontally and take threshold and then 
                # seperate the image
                binarized_img = []
                for ii , imag in enumerate(data_img):
                    if ii == 0 :
                        imag_ext = np.concatenate((imag, data_img[-1]), axis=1, dtype=np.uint8)
                        ret, ch, con = self.otsu_binarize(imag_ext, bin_type)
                        ret = np.split(ret, [imag_ext.shape[1]//2], axis=1)[0]
                        ret = cv2.cvtColor(ret, cv2.COLOR_GRAY2BGR)
                        ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
                        binarized_img.append((ret, ch, con))
                    else:
                       binarized_img.append( self.otsu_binarize(imag, bin_type) )
                        
                self.threshold = [im[1] for im in binarized_img]
                binarized_img = [im[0] for im in binarized_img]
                self.binarized_img = binarized_img
                self.update_th_label()
    
    
            else:
                binarized_img = [self.custom_binarize(image, bin_type
                                                      ) for image in data_img]
                self.threshold = [im[1] for im in binarized_img]
                binarized_img = [im[0] for im in binarized_img]
                self.binarized_img = binarized_img
                self.update_th_label()
            
            self.load_images(binarized_img)
            
            del data_img, binarized_img  
        else:
            
            self.binarized_img = None
            self.load_images(self.images)
            self.update_th_label()
            
          
        self.tabWidget.setCurrentIndex(current_index)
        
        
        
    def qimage_fromdata(self, img):
        '''define a function to convert NumPy array to QImage'''
        
        height, width = img.shape
        qimg = QImage(img.data, width, height, QImage.Format_Grayscale8)
        
        return qimg
    

    def qimage_to_numpy(self, qimage): 
        '''define a function to convert QImage to NumPy array'''
        
        width = qimage.width() # get the width of the QImage
        height = qimage.height() # get the height of the QImage
        # get the number of bytes per line of the QImage
        bytes_per_line = qimage.bytesPerLine() 
        # get the raw data of the QImage as a NumPy array
        image_data = qimage.bits().asarray(bytes_per_line * height) 
        # reshape and cast the array to match the grayscale image format
        return np.reshape(image_data, (height, width)).astype(np.uint8)
    
    # This slot will be called when the close button of a tab is pressed
    def on_tab_close_requested(self,index):
        self.tabWidget.widget(index).deleteLater()
        self.tabWidget.removeTab(index)
        
    def move_tab(self, direction):
        '''move to the next card in the direction defined'''
        
        current_index = self.tabWidget.currentIndex()
        
        if direction == 'right':
            next_index = (current_index + 1) % self.tabWidget.count()
            self.tabWidget.setCurrentIndex(next_index)
        elif direction == 'left':
            next_index = (current_index - 1) % self.tabWidget.count()
            self.tabWidget.setCurrentIndex(next_index)
            
    def otsu_binarize(self, image, binarize_type):
        """
        take an image path as input and give binarized image, threshold image, contour as output
    
        Parameters
        ----------
        image : np.array
            path of the image.
    
        Returns
        -------
        ret : TYPE
            the input image GRAY.
        th : TYPE
            threshold image - OTSU_Gasussian Threshold.
        contour : TYPE
            return the contours in the image.
    
        """
        img = image
        img_gus = cv2.GaussianBlur(img, (25,25),0)
        if binarize_type.inverse:
            
            th, ret = cv2.threshold(img_gus, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
        else :
            
            th, ret = cv2.threshold(img_gus, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            
        contour, hei = cv2.findContours(ret, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # ret = cv2.Canny(ret, 0, 255, 7)
    
        return (ret, th , contour)
    
    def custom_binarize(self, image, binarize_type):
        """
        take an image , threshold image, contour as output
    
        Parameters
        ----------
        image : np.array
            path of the image.
    
        Returns
        -------
        ret : TYPE
            the input image GRAY.
        th : TYPE
            threshold image - OTSU_Gasussian Threshold.
        contour : TYPE
            return the contours in the image.
    
        """

    
        img = image
        img_gus = cv2.GaussianBlur(img, (25, 25),0)
        if binarize_type.inverse:
    
            th, ret = cv2.threshold(img_gus, binarize_type.threshold, 255, cv2.THRESH_BINARY_INV)
            
        else :
            
            th, ret = cv2.threshold(img_gus, binarize_type.threshold, 255, cv2.THRESH_BINARY)
            
        contour, hei = cv2.findContours(ret, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # ret = cv2.Canny(ret, 0, 255, 7)
        
        return (ret, th , contour)
    
    def edge_detection(self, method):
        
        ret = [cv2.Canny(img, 0, 255, 7) for img in self.binarized_img]
        
    def def_direction(self):
        
        tab = self.tabWidget.widget(self.tabWidget.currentIndex())
        tabs = [self.tabWidget.widget(i) for i in range(self.tabWidget.count()) if self.tabWidget.widget(i)]
        
        if tab.first_click is not None and tab.second_click is not None:
            
            self.linep1_x.setValue(tab.first_click.x())
            self.linep1_y.setValue(tab.first_click.y())
            self.linep2_x.setValue(tab.second_click.x())
            self.linep2_y.setValue(tab.second_click.y())
            
        else :
            
            self.linep1_x.setValue(0)
            self.linep1_y.setValue(0)
            self.linep2_x.setValue(0)
            self.linep2_y.setValue(0)
    
    def set_direction(self):
        
        tab = self.tabWidget.widget(self.tabWidget.currentIndex())
        tabs = [self.tabWidget.widget(i) for i in range(self.tabWidget.count()) if self.tabWidget.widget(i)]        
       
        # taking the points cordinates form input
        p1 =  QPoint(self.linep1_x.value(), self.linep1_y.value())
        p2 = QPoint(self.linep2_x.value(), self.linep2_y.value())
        
        # checking check box if it has to be set for all tabs
        if not self.checkBox.isChecked():
            
            # if any of the inputs are non zero line should be redrawn
            if p1 and p2:
                
                tab.set_line_points(p1, p2)
                
            # if all the inputs are zero line should be deleted    
            else:
                
                tab.set_line_points(None, None)
        # applying the logic to all tabs if needed.
        else :
            
            if p1 and p2:
                
                [tab.set_line_points(p1, p2) for tab in tabs]
                
            else:
                
                [tab.set_line_points(None, None) for tab in tabs]
            

    def outline_dial(self, number):
        '''
        Changes the outline in the line drawn on Qlabel

        Parameters
        ----------
        number : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        tabs = [self.tabWidget.widget(i) for i in range(self.tabWidget.count()) if self.tabWidget.widget(i)]
        [tab.set_outline(number) for tab in tabs]
    
    def binarize(self, image):
        
        bin_type = ps.Binarize_Type.from_window(self)
        
        if bin_type == 1:
            
            if np.all(image == self.images[0]):
                
                imag_ext = np.concatenate((image, self.images[-1]), axis=1, dtype=np.uint8)
                ret, ch, con = self.otsu_binarize(imag_ext, bin_type)
                ret = np.split(ret, [imag_ext.shape[1]//2], axis=1)[0]
                ret = cv2.cvtColor(ret, cv2.COLOR_GRAY2BGR)
                ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
                img_plus = ( ret, ch, con )
            else:
         
                binarize = self.otsu_binarize
                img_plus = binarize(image, bin_type)
        else:
            binarize = self.custom_binarize
            img_plus = binarize(image, bin_type)
            
        return img_plus
            
            
    def add_edge(self):
        
        # if self.b_combo_box.currentIndex() == 1:
        #     binarize = self.otsu_binarize
        # else:
        #     binarize = lambda image : self.custom_binarize(image, self.spinBox.value())
            
        bin_imgs = [self.binarize(image)[0] for image in self.images]
        
        # creating a ne Meas_Type object. Advantage of this object is that if its bubble type domain
        # its easy to pass the center to the bdw_detect function
        measurmentType = Meas_Type(window = self)    
        
        # selecting according to the type of measurements
        if measurmentType == 0:
            
            # if bubble domain type detect the closed contour corresponding to the domain using ps.bdw_detect
            dws = [ps.bdw_detect(image, measurmentType.center) for image in bin_imgs]
            # ploting the contour ie the domain wall to a black image of same shape
            new_img = ps.image_from_coords(dws[0], self.images[0].shape)
            [ps.image_from_coords(dw, self.images[0].shape,new_img) for dw in dws]
            
            # the output of ps.bdw_detect is a 2d array but contours are represented as 3d array with axis 1 has one 
            # so we are converting it into a 3d array for to be used by ps.mark_center
            dws_contours = [np.expand_dims(array, axis=1) for array in dws]
            # making the center of each contour in the image
            [ps.mark_center(contour, new_img) for contour in dws_contours]
            
        elif measurmentType == 1:
            
            
            # if domain of random shape then it probably has a psedo open contour so using ps.dw_detect which
            # uses a edege detection
            dws = [ps.dw_detect(image) for image in bin_imgs]
            new_img = ps.image_from_coords(dws[0], self.images[0].shape)
            [ps.image_from_coords(dw, self.images[0].shape,new_img) for dw in dws]
   
            
            
        # print(dws)    
        
        
        new_img[new_img > 0] = 255
        new_img = new_img.astype(np.uint8)
        # new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
        
        tab = MyLabel()
        
        # tab.setScaledContents(True)
        # tab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        image = self.qimage_fromdata(new_img)
        tab.resize(image.size().width(), image.size().height())
        tab.setMaximumSize(image.size().width(), image.size().height())
        
        tab.setPixmap(QPixmap.fromImage(image))
        
        # tab.setMask(QPixmap.fromImage(image).mask())
        # cv2.imshow('Image 2', new_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # plt.imshow(new_img)

        self.tabWidget.addTab(tab, 'edges')
        
    def calculate_motion(self, images = None):
        
        # point2 = ps.Point(self.linep1_x.value(),self.linep1_y.value())
        # point1 = ps.Point(self.linep2_x.value(), self.linep2_y.value())
        # outline =  self.dial.value()
        if images:
            bin_imgs = [self.binarize(image)[0] for image in images]
        
        else:
            bin_imgs = [self.binarize(image)[0] for image in self.images]
            
        # creating a ne Meas_Type object. Advantage of this object is that if its bubble type domain
        # its easy to pass the center to the bdw_detect function
        measurmentType = Meas_Type(window = self)
        point2 = measurmentType.point2
        point1 = measurmentType.point1
        outline = measurmentType.outline
        # selecting according to the type of measurements
        if measurmentType == 0:
            
            # if bubble domain type detect the closed contour corresponding to the domain using ps.bdw_detect
            dws = [ps.bdw_detect(image, measurmentType.center) for image in bin_imgs]
            
        elif measurmentType == 1:
            
            
            # if domain of random shape then it probably has a psedo open contour so using ps.dw_detect which
            # uses a edege detection
            dws = [ps.dw_detect(image) for image in bin_imgs]
    
        motion = ps.Domain_mot(dws)
        
        lines = ps.parallel_lines(point2, point1, outline)
        
        distance = [motion.distance(line) for line in lines]

        # distance = motion.distance([point2, point1,])
        if not images:
            print(distance)
        return distance
        
        
    def dt_curve(self, paths : pathlib.Path, voltage : str) -> pd.DataFrame:
        '''
        Takes the parent path which contains the voltage measurements and returns 
        a displacement v/s pulse_width data
        
        Parameters
        ----------
        path : pathlib.Path
            Parent path which contain the voltage folders.
        voltage : str or int
            Starting of the voltage folder names. For example if the folders are 25V,25V_1
            25V_2, 25V_3 etc then 25 or 25V can be given as voltage.
        
        Returns
        -------
        dta1 : Pandas DataFrame
            Returns displacement v/s pulse_width data.
        
        '''
            
        dta = pd.DataFrame(columns = ['pulse_width', 'displacement'])
        
        for path in list(paths.glob(f'{voltage}*')):
            # print(path)
            images = [*path.glob('*.png')]
            pulse_width = ps.get_pwidth(images[0])[0]
            images = [cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)[:512] for image in images]
            displacement = self.calculate_motion(images)
            displacement = [x for x in displacement if x]
            displacement = np.mean([np.mean(dis) for dis in displacement])
            
            dta.loc[len(dta)] = [pulse_width, displacement]
        dta1 =  pd.DataFrame(columns = ['pulse_width', 'displacement'])
        dta['pulse_width'] = np.round(dta['pulse_width'],6)
        dta =  dta[np.logical_not(np.isnan(dta['displacement']))]
        for x in set(dta['pulse_width']):
            if not np.isnan( x ):
                
                #dataframe.append will be deprecated in future pandas so going to us concat insted
                #dta1 = dta1.append(dta[dta['pulse_width']==x].mean(),ignore_index=True)
                dta1 = pd.concat([dta1,dta[dta['pulse_width']==x].mean().to_frame().T],ignore_index=True)
        dta1.sort_values(by = 'pulse_width', inplace = True)
        dta1.reset_index(drop = True, inplace = True)
        return dta1

        
    def calculate_all(self):
        
        paths = pathlib.Path(self.textInputFolder.toPlainText()).parent
        # out_path = paths/"DisplacementTime"
        out_path = paths/f"{self.save_folder.text()}"
        
        if not out_path.exists():
            out_path.mkdir()
        else:
            # if the overwrite checkbok is not checked it will skip this directory
            if not self.bulk_overwrite_check.isChecked():
                QMessageBox(QMessageBox.Warning, 
                            "Warning", 
                            f"Specified Folder '{self.save_folder.text()}' Already Exists in the directory'{paths.name}'",
                            parent= self).exec()
                return None
        
        volts= ps.find_volt(paths)
            
        arguments = dict(
        measurmentType = Meas_Type(window = self),
        binarize = ps.Binarize_Type.from_window(window = self))

            
        do_partial =  partial(do,paths = paths, out_path = out_path, **arguments)    
        with Pool(4) as p:

            results = p.map(do_partial, volts)    
        
        # for x in volts:
        
        #     out = ps.float_str(ps.field(float(x)/10),2) + 'mT.txt'
            
        #     dat = self.dt_curve(paths, x)
            
        #     dat.to_csv(out_path/out, sep = '\t', index = False)
            
    def calculate_all_from_parent(self):
        
        paths_parent = pathlib.Path(self.textInputFolder.toPlainText()).parent.parent
        
        folder_name = self.save_folder.text()
        

        
        arguments = dict(
        measurmentType = Meas_Type(window = self),
        binarize = ps.Binarize_Type.from_window(window = self))
        
        iter_path = [path for path in paths_parent.iterdir()]
        
        for paths in iter_path:
            
            
            if paths.is_dir() and paths.name.endswith('mT'):
                
                # out_path = paths/"DisplacementTime"
                out_path = paths/folder_name
                
                if not out_path.exists():
                    out_path.mkdir()
                else:
                    # if the overwrite checkbok is not checked it will skip this directory
                    if not self.bulk_overwrite_check.isChecked():
                        msg_box = QMessageBox(QMessageBox.Warning, 
                                    "Warning", 
                                    f"Specified Folder '{folder_name}' Already Exists in the directory'{paths.name}'" ,
                                    parent = self)
                        msg_box.setModal(False)
                        msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
                        msg_box.show()
                        continue
                        
                print(paths)
                
                volts= ps.find_volt(paths)
                
                do_partial =  partial(do,paths = paths, out_path = out_path, **arguments)
                
                with Pool(4) as p:

                    results = p.map(do_partial, volts)    
                    
                    
    def plot(self,path):
        '''
        Plot the data in the current save folder
        Parameters
        ----------
        path : str|pathlib.Path
            path of the dir where the files are.

        Returns
        -------
        None.

        '''
        pattern = re.compile(r"\d+\.\d+mT\.txt")
        
        
        path = pathlib.Path(path)
        
        files = [x for x in path.iterdir() if pattern.match(x.name)]
        
        fig, ax = plt.subplots()
        # sorting the files according to the field value
        files.sort(key = lambda x: float(x.name.split('m')[0]))
        
        # to read the comment character from the first file
        
        comment_char = "#"
        comments = [] # it reads all the comment lines from the file if avilable
        with open(files[0], "r") as f:
            for line in f:
                if line.startswith(comment_char):
                    comments.append(line.strip())
                    
        
        if comments:
            # we only need the comment in the first line
            string= comments[0].replace('# ', '').replace(',  ', '\n')            
        else:            
            mt = Meas_Type(window = self)
            bi = ps.Binarize_Type.from_window(window = self)
            
            string = f'''point2 : {mt.point2}
point1 : {mt.point1}
DCenter : {mt.center}
Binarize : {bi.threshold}
outline : {mt.outline}'''


        ax.text(0.25, 0.85, string, transform=ax.transAxes)
            
        
        for file in files:
            
            data = pd.read_csv(file, 
                               sep = '\t',
                               comment = comment_char
                               )
            if data.empty:
                print(f"{str(file)} does not contain any data")
                continue
            data.plot(x = 0, y = 1 , style = 'o', label = file.name.split('.txt')[0], ax = ax)
            



        plt.show()
        

class Meas_Type():
    
    def __init__(self, window):
        
        self.index = window.measurmentType.currentIndex()
        self.center = window.center_select.text()
        if self.center:
            self.center = ps.Point(*map(int, self.center.split(",")))
            
        # point2 is the first click and point1 is second click        
        self.point2 = ps.Point(window.linep1_x.value(), window.linep1_y.value())
        self.point1 = ps.Point(window.linep2_x.value(), window.linep2_y.value())
        self.outline =  window.dial.value()
        
    def __eq__(self, other):
        
        return self.index == other
    

 #%%   
if __name__ == '__main__':
    multiprocessing.freeze_support()
    # create an application object
    app = QApplication(sys.argv)
    # create an example object
    window = MainWindow()
    window.show()
    # [k for k in dir(window) if k not in dir(QMainWindow) and not k.startswith('__')]
    # exit the application when the window is closed
    
    # sys.exit(app.exec_())
    app.exec_()    
