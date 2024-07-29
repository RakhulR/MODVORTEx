# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:24:16 2023

@author: Rakhul Raj
"""
#%%
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
from custom_widgets import MyLabel, SettingWindow
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


def do(volt, paths, out_path, measType, binarize):
    
    out = ps.float_str(ps.field(float(volt)/10),2) + 'mT.txt'
    
    dat = ps.dt_curve(paths, volt, measType, binarize)
    
    mt = measType
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
        self.dial.valueChanged.connect(self.dial_changed)
        self.genEdges.clicked.connect(self.add_edge)
        self.calcButton.clicked.connect(lambda : print(
                                                        ps.calculate_motion_displace(self.images, ps.Meas_Type.from_window(self), 
                                                                         ps.Binarize_Type.from_window(self)
                                                                         )
                                                        )
                                        )
        self.calcAllButton.clicked.connect(self.calculate_all)
        self.bulkcalcButton.clicked.connect(self.calculate_all_from_parent)
        
            
        self.loadfolder.clicked.connect(self.loadfolder_f)
        self.plot_button.clicked.connect(lambda  : self.plot(Path(self.textInputFolder.toPlainText()).parent/
                                                             self.save_folder.text()                                                             
                                                             ))
        
        self.dselect_button.clicked.connect(self.auto_domain_select)
        self.plt_histogram_b.clicked.connect(self.plt_hist)
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

        self.settings_win = SettingWindow(self)

        self.load_settings.clicked.connect(self.settings_win.show)
        self.load_settings.clicked.connect(self.settings_win.activateWindow)
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
        
            # images = [cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)[:512] for image in images]
            measType =  ps.Meas_Type.from_window(self)
            images = [ps.load_image(image, measType) for image in images]
            self.images = images
            images = [self.qimage_fromdata(image) for image in images]
            
            tabs = [MyLabel(mainwindow= self) for _ in images]
            [x.set_outline(self.dial.value()) for x in tabs]
            [label.resize(image.size().width(), image.size().height()) for label, image in zip(tabs,images)]
            [label.setMaximumSize(image.size().width(), image.size().height()) for label, image in zip(tabs,images)]
            [tab.setPixmap(QPixmap.fromImage(image))for tab, image in zip(tabs,images)]
            
            for ii, tab in enumerate(tabs):
                
                self.tabWidget.addTab(tab, f"tab{ii}")
                

        else :
            
            images = [self.qimage_fromdata(image) for image in update_images]
            
            tabs = [self.tabWidget.widget(i) for i in range(self.tabWidget.count()) if self.tabWidget.widget(i)]
            
            [tab.setPixmap(QPixmap.fromImage(image))for tab, image in zip(tabs,images)]

    def auto_domain_select(self):

        binarize = ps.Binarize_Type.from_window(self)
        bin_imgs = binarize.binarize_list(images= self.images)

        measType = ps.Meas_Type.from_window(window = self)
        measType.index = ps.Meas_Type.BUBBLE_CIRCLE_FIT
        measType.select_domain = False
        motions = ps.cexpand_detect(bin_imgs, measType)
        # print([x.centre for x in motions])
        centres = [str(x.centre).replace(' ', '') for x in motions]
        # print(centres)
        centres_str = ';'.join(centres)

        self.select_domain_line.setText(centres_str) 



    def dial_changed(self, value):

        # change the label in qlabel
        self.label_dial.setText(f'Width\n({value})')

        # Changes the outline in the line drawn on all the custom labels
        tabs = [self.tabWidget.widget(i) for i in range(self.tabWidget.count()) if self.tabWidget.widget(i)]
        [tab.set_outline(value) for tab in tabs]

    def update_position_label(self, x:str, y:str):
        ''' Updates the position of mouse in MyLabel'''
        self.coord_label.setText(f"Current Coordinates (pixels): ({x}, {y})")  # Update with the current coordinates

    def plt_hist(self):

        kernel = ps.Binarize_Type.from_window(window= self).kernel

        ind = self.tabWidget.currentIndex()
        if ind < len(self.images): # current tabs are image tabs
            ps.plt_histogram(image= self.images[ind],threshold= self.spinBox.value(), blur_kernel= kernel )
        else: # current tab is edge tab
        
            label: QLabel = self.tabWidget.widget(ind)
            
            pixmap: QPixmap = label.pixmap()

            if pixmap:
                qimage : QImage = pixmap.toImage()
                image : np.array = ps.qimage_to_array(qimage= qimage)
                if len(image.shape) == 3:
                    if image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        ps.plt_histogram(image= image, blur_kernel= kernel)
                    elif image.shape[2] == 1:
                        ps.plt_histogram(image= image.squeeze(), blur_kernel= kernel)
                    else:
                        raise ValueError(f"The image format is not recognized as RBG or Grayscale. shape is {image.shape}")
                elif len(image.shape) == 2:
                    ps.plt_histogram(image= image, blur_kernel= kernel)
                else:
                    raise ValueError(f"The image format is not recognized as RBG or Grayscale. shape is {image.shape}")
            else:

                raise RuntimeError("No pixmap available in the current tab")
        
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
            if  bin_type == ps.Binarize_Type.TYPE_OTSU:
                
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
        # qimg = QImage(img.data, width, height, QImage.Format_Grayscale8) # error was happening when i crop the width of the image
        qimg = QImage(bytes(img.data), width, height, QImage.Format_Grayscale8)
        
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
            
    def otsu_binarize(self, image, binarize_type: ps.Binarize_Type):
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
        img_gus = cv2.GaussianBlur(img, binarize_type.kernel, 0)
        if binarize_type.inverse:
            
            th, ret = cv2.threshold(img_gus, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
        else :
            
            th, ret = cv2.threshold(img_gus, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            
        contour, hei = cv2.findContours(ret, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # ret = cv2.Canny(ret, 0, 255, 7)
    
        return (ret, th , contour)
    
    def custom_binarize(self, image, binarize_type: ps.Binarize_Type):
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
        img_gus = cv2.GaussianBlur(img, binarize_type.kernel, 0)
        if binarize_type.inverse:
    
            th, ret = cv2.threshold(img_gus, binarize_type.threshold, 255, cv2.THRESH_BINARY_INV)
            
        else :

            th, ret = cv2.threshold(img_gus, binarize_type.threshold, 255, cv2.THRESH_BINARY)
            
        contour, hei = cv2.findContours(ret, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # ret = cv2.Canny(ret, 0, 255, 7)
        
        return (ret, th , contour)
    
        
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

      
    def add_edge(self):
        
        # if self.b_combo_box.currentIndex() == 1:
        #     binarize = self.otsu_binarize
        # else:
        #     binarize = lambda image : self.custom_binarize(image, self.spinBox.value())
        binarize = ps.Binarize_Type.from_window(self)
        bin_imgs = binarize.binarize_list(images= self.images)
        
        # creating a ne Meas_Type object. Advantage of this object is that if its bubble type domain
        # its easy to pass the center to the bdw_detect function
        measType = ps.Meas_Type.from_window(window = self)

        if measType == ps.Meas_Type.BUBBLE_CIRCLE_FIT:

            motions = ps.cexpand_detect(bin_imgs, measType)
            shape = self.images[0].shape
            new_img = ps.image_from_coords(np.array([], dtype = 'int32').reshape(0, 0),shape) # Look here
            for motion in motions:
                [ps.image_from_coords(domain.contour.squeeze(), shape, new_img)  
                 for domain in motion.domains]

        # selecting according to the type of measurements
        elif measType == ps.Meas_Type.BUBBLE_DIRECTIONAL:
            
            # if bubble domain type detect the closed contour corresponding to the domain using ps.bdw_detect
            dws = [ps.bdw_detect(image, measType.center) for image in bin_imgs]

            shape = self.images[0].shape
            # ploting the contour ie the domain wall to a black image of same shape
            new_img = ps.image_from_coords(dws[0], shape)
            [ps.image_from_coords(dw, shape, new_img) for dw in dws]
            
            # the output of ps.bdw_detect is a 2d array but contours are represented as 3d array with axis 1 has one 
            # so we are converting it into a 3d array for to be used by ps.mark_center
            dws_contours = [np.expand_dims(array, axis=1) for array in dws]
            # making the center of each contour in the image
            [ps.mark_center(contour, new_img) for contour in dws_contours]
            
        elif measType == ps.Meas_Type.ARBITARY_STRUCTURE:
            
            # if domain of random shape then it probably has a psedo open contour so using ps.dw_detect which
            # uses a edege detection
            dws = [ps.dw_detect(image) for image in bin_imgs]
            shape = self.images[0].shape
            new_img = ps.image_from_coords(dws[0], shape)
            [ps.image_from_coords(dw, shape, new_img) for dw in dws]
   
            
            
        # print(dws)    
        
        
        new_img[new_img > 0] = 255
        new_img = new_img.astype(np.uint8)
        # new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
        
        tab = MyLabel(mainwindow= self)
        tab.set_outline(self.dial.value()) # setting the outline value of mylabel
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
                        measType = ps.Meas_Type.from_window(window = self),
                        binarize = ps.Binarize_Type.from_window(window = self)
                        )

            
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
                        measType = ps.Meas_Type.from_window(window = self),
                        binarize = ps.Binarize_Type.from_window(window = self)
                        )
        
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
            mt = ps.Meas_Type.from_window(window = self)
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
        
    # def closeEvent(self, event):
    #     if self.settings is not None:
    #         self.settings.close()
    #     event.accept()

 #%%   
if __name__ == '__main__':
    multiprocessing.freeze_support()
    # create an application object
    app = QApplication(sys.argv)
    # create a mainwindow
    window = MainWindow()
    window.show()
    # [k for k in dir(window) if k not in dir(QMainWindow) and not k.startswith('__')]
    # exit the application when the window is closed
    
    # sys.exit(app.exec_())
    app.exec_()    
