# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:01:56 2023

@author: Rakhul Raj
"""

import sys
import os
import re
import concurrent.futures
from multiprocessing import Pool
from threading import Thread
from functools import partial
from pathlib import Path

DOCS_URL = "https://github.com/RakhulR/MODVORTEx"
from typing import TYPE_CHECKING, Union
from typing import TypeVar, Tuple, Any, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PyQt5.QtGui import QMouseEvent, QPainter, QPen
from PyQt5.QtWidgets import (QMainWindow,
                             QApplication,
                             QMessageBox,
                             QFileDialog,
                             QLabel,
                             QShortcut,
                             QSizePolicy,
                             QWidget,
                             QPushButton,
                             QDialog,
                             QScrollArea,
                             QVBoxLayout,
                             QHBoxLayout,
                             QListWidget,
                             QListWidgetItem,
                             QDialogButtonBox,
                             QAction
    )
from PyQt5 import uic
from PyQt5.QtCore import QSize, QPoint, Qt, QUrl
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QDesktopServices
# iconSize = QSize(3000,3000)
# from werkzeug.serving import make_server

from mymodule.utils import decorate_all_methods
from mymodule.exceptions import exception_handler
import icons
import processing as ps
from api import MyFlaskApp
import imageviewer

# if TYPE_CHECKING:
from multiprocessing.pool import Pool as MPPool
if sys.version_info >= (3, 9):

    # Define custom types
    GrayImage = TypeVar('GrayImage', bound= np.ndarray[Tuple[Any, Any], np.uint8])
    BGRImage = TypeVar('BGRImage', bound= np.ndarray[Tuple[Any, Any, 3], np.uint8])
    Image = TypeVar('Image', bound= np.ndarray[Tuple[Any, Any, 0|3], np.uint8])
    Contour = TypeVar('Contour', bound= np.ndarray[Tuple[Any, 1, 2], np.uint8])
    # end
else:
    # Define custom types
    GrayImage = TypeVar('GrayImage', bound= np.ndarray)
    BGRImage = TypeVar('BGRImage', bound= np.ndarray)
    Image = TypeVar('Image', bound= np.ndarray)
    Contour = TypeVar('Contour', bound= np.ndarray)

if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    application_path = Path(sys._MEIPASS)
else:
    application_path = Path(os.path.dirname(os.path.abspath(__file__)))

class PoolManager:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.pool : Union[MPPool, None] = None

    def start(self):
        if self.pool is None:
            self.pool = Pool(self.num_workers)

        return self.pool

    def wait(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def terminate(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool = None

def do(volt, paths, out_path, measType, binarize):

    sel_paths: List[Path] = [dirs for dirs in paths.iterdir() if re.match(rf'^{volt}\D+', dirs.name) and dirs.is_dir()]
    sel_paths.sort(key = ps.sort_key)
    
    # extracting the unit of the folders in which images are found
    unit = re.findall(r'[^0-9.]+', sel_paths[0].name)
    unit = unit[0] if unit else ""
    unit = unit[:-1] if unit.endswith('_') else unit 

    # defining the output filename
    out =  volt + f"{unit}.txt"
    
    dat = ps.dt_curve(sel_paths, volt, measType, binarize)
    
    mt = measType
    bi = binarize
    
    string = f'# point2 : {mt.point2},  point1 : {mt.point1},  DCenter : {mt.center},  Binarize : {bi.threshold},\
outline : {mt.outline}\n'
        
    with open(out_path/out, 'w') as f:
        f.write(string)
    dat.to_csv(out_path/out, mode = 'a',  sep = '\t', index = False)


class Worker(imageviewer.Worker):

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self.mainwindow = window.mainwindow
    
    # @exception_handler
    def get_state(self):

        while self._is_running:
            if self.window.process_box.isChecked():
                try:

                    state = {
                        'measType': ps.Meas_Type.from_window(self.mainwindow),
                        'binarize': ps.Binarize_Type.from_window(self.mainwindow)
                    }
                except Exception as exp:
                    print(exp)
                    continue

                self.state_updated.emit(state)
                imageviewer.time.sleep(0.2)
            else:
                imageviewer.time.sleep(0.5)
                
@decorate_all_methods(exception_handler)
class Modvortex_ImageProcessor(imageviewer.ImageProcessor):
    def __init__(self, parent: QMainWindow, worker= Worker):
        self.mainwindow = parent
        super().__init__(worker)
        # self.measType = ps.Meas_Type.from_window(self.mainwindow)

        # getting the start image from mainwindow and setting it
        text = self.mainwindow.input_folder()
        self.start_path = Path(text)
        self.set_folder(self.start_path)
    
    
    def load_images(self):
        super().load_images()

        if self.folders:
            # changing the folder and images in the mainwindow
            folder = self.folders[self.current_folder_index]
            self.mainwindow.textInputFolder.setPlainText(str(folder))
            self.mainwindow.load_images(self.images)
            self.mainwindow.images = self.images


    def closeEvent(self, event):
        # resetting the folder in the folder box in mainwindow
        self.mainwindow.textInputFolder.setPlainText(str(self.start_path))
        self.mainwindow.load_images()
        super().closeEvent(event)
        
    

class ReorderDialog(QDialog):
    """Let the user confirm/arrange image time order with a preview.

    Shown when filenames carry no coded pulse width, so the acquisition
    order cannot be derived from the names alone. The list is pre-filled
    with the predicted (natural-number) order; the first row acts as the
    reference image.
    """

    def __init__(self, parent=None, file_names=None, preview_loader=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm image order")
        self.setWindowModality(Qt.WindowModal)
        self.resize(560, 420)
        self._preview_loader = preview_loader

        info = QLabel(
            "The program could not determine the time order from the file "
            "names.\nPlease arrange the images oldest-first (first row = "
            "reference image) and press OK."
        )
        info.setWordWrap(True)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(info)

        content = QHBoxLayout()
        main_layout.addLayout(content)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SingleSelection)
        # Both drag-and-drop and Up/Down buttons are supported.
        self.file_list.setDragDropMode(QListWidget.InternalMove)
        self.file_list.setDefaultDropAction(Qt.MoveAction)
        for name in (file_names or []):
            QListWidgetItem(name, self.file_list)
        if self.file_list.count():
            self.file_list.setCurrentRow(0)
        content.addWidget(self.file_list, 1)

        side = QVBoxLayout()
        content.addLayout(side)

        self.preview_label = QLabel("Select an image to preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(280, 280)
        self.preview_label.setScaledContents(False)
        side.addWidget(self.preview_label)

        move_row = QHBoxLayout()
        side.addLayout(move_row)
        self.up_button = QPushButton("Move Up")
        self.down_button = QPushButton("Move Down")
        move_row.addWidget(self.up_button)
        move_row.addWidget(self.down_button)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        main_layout.addWidget(self.button_box)

        self.up_button.clicked.connect(self.move_up)
        self.down_button.clicked.connect(self.move_down)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.file_list.currentRowChanged.connect(self.update_preview)
        self.update_preview(self.file_list.currentRow())

    def move_up(self):
        row = self.file_list.currentRow()
        if row > 0:
            item = self.file_list.takeItem(row)
            self.file_list.insertItem(row - 1, item)
            self.file_list.setCurrentRow(row - 1)

    def move_down(self):
        row = self.file_list.currentRow()
        if 0 <= row < self.file_list.count() - 1:
            item = self.file_list.takeItem(row)
            self.file_list.insertItem(row + 1, item)
            self.file_list.setCurrentRow(row + 1)

    def update_preview(self, row):
        if self._preview_loader is None or row is None or row < 0:
            return
        try:
            pixmap = self._preview_loader(self.file_list.item(row).text())
        except Exception:
            pixmap = None
        if pixmap is None or pixmap.isNull():
            self.preview_label.setText("No preview available")
            self.preview_label.setPixmap(QPixmap())
        else:
            self.preview_label.setPixmap(
                pixmap.scaled(
                    self.preview_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def ordered_names(self):
        return [
            self.file_list.item(i).text()
            for i in range(self.file_list.count())
        ]


@decorate_all_methods(exception_handler)
class MainWindow(QMainWindow):
    
    def __init__(self):
        
        super(MainWindow, self).__init__()
        uic.loadUi(application_path/"mainwindow_v_new.ui", self)
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
        self.genEdges.clicked.connect(self.set_edge)
        self.calcButton.clicked.connect(self.calculate_current)
        self.calcAllButton.clicked.connect(lambda :  self.calculate_all()
                                           if self.calcAllButton.text() == "Calculate All"
                                            else self.stop_calculate_all() )
        self.bulkcalcButton.clicked.connect(self.calculate_all_from_parent)
        # Bulk Calculate stays disabled until explicitly enabled in the future.
        self.bulkcalcButton.setEnabled(False)
        
            
        self.loadfolder.clicked.connect(self.loadfolder_f)
        self.plot_button.clicked.connect(lambda  : self.plot(Path(self.input_folder()).parent/
                                                             self.result_folder_name()
                                                             ))
        
        self.dselect_button.clicked.connect(self.auto_domain_select)
        self.show_domain_fit.clicked.connect(self.show_circle_fit)
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

        self.img_viewer = None


        self.settings_win = SettingWindow(self)


        self.actionAbout.triggered.connect(self.show_about_dialog)
        self.actionSettings.triggered.connect(self.settings_win.show)
        self.actionSettings.triggered.connect(self.settings_win.activateWindow)
        self.actionImage_Viewer.triggered.connect(self.open_img_viewer)
        self.actionSelect_Folder.triggered.connect(self.loadfolder_f)
        self.actionDocumentation.triggered.connect(self.open_docs)

        # images that are loaded in the tabs
        self.images = None
        # paths of the images backing the tabs, in tab order
        self._loaded_paths = []
        # manually confirmed orders per folder: {folder: [file names]}
        self._manual_orders = {}
        # save the threshold value while binarizing
        self.threshold = None

        # action to re-open the manual ordering dialog for loaded images
        self.actionReorderImages = QAction("Reorder Images", self)
        self.actionReorderImages.triggered.connect(self.reorder_current_images)
        if hasattr(self, 'menuEdit'):
            self.menuEdit.addAction(self.actionReorderImages)
        
        # set the file filter to show only .py files
        # dialog.setNameFilter("Python files (*.png)")

        # poolmanager for calculating function

        self.pool = PoolManager(num_workers= 4 )



        # for flask api
        self.flask_app = MyFlaskApp(self)
        self.api_button.clicked.connect(self.toggle_api)

    def show_about_dialog(self):
        version = QApplication.instance().applicationVersion()
        QMessageBox.about(self, "About", f"""<p>MODVORTEx<br>
        (Magneto Optical Domain Velocity Observation and Real-Time Extraction)<br>
        Version: {version}<br>
        Written By Rakhul Raj<br>
        If this program is helpful in your work, please cite our articles <a href="https://dx.doi.org/10.1088/1361-6501/ad8beb">1</a> and <a href="https://dx.doi.org/10.1088/1361-6501/acd39d">2</a>.</p>""")


    def open_docs(self):
        QDesktopServices.openUrl(QUrl("https://github.com/RakhulR/MODVORTEx"))


    def input_folder(self) -> str:
        """Stripped folder path from the path box (writes the cleanup back)."""
        raw = self.textInputFolder.toPlainText()
        cleaned = raw.strip()
        if cleaned != raw:
            self.textInputFolder.setPlainText(cleaned)
        return cleaned

    def result_folder_name(self) -> str:
        """Stripped result save-folder name (writes the cleanup back)."""
        raw = self.save_folder.text()
        cleaned = raw.strip()
        if cleaned != raw:
            self.save_folder.setText(cleaned)
        return cleaned

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
            res = Path(folder[0]).parent
            self.textInputFolder.setPlainText(str(res))


    # Helper: get the label inside a tab (supports QScrollArea or direct label)
    def get_label_from_tab(self, index):
        w = self.tabWidget.widget(index)
        if w is None:
            return None
        # If the tab content is a QScrollArea, the label is its widget
        if isinstance(w, QScrollArea):
            return w.widget()
        # If the tab content is a container, try to find the first QLabel-like child
        # (your MyLabel derives from QLabel and supports setPixmap)
        for child in w.findChildren(QWidget):
            # Prefer the first MyLabel or QLabel found
            if hasattr(child, 'setPixmap'):
                return child
        # If the tab content itself is the label
        if hasattr(w, 'setPixmap'):
            return w
        return None

    def add_label_as_tab(self, label: "MyLabel", title: str):
        """
        Wrap a given MyLabel in a scrollable container and add it as a tab.

        Parameters
        ----------
        label : MyLabel
            The custom label containing the image/pixmap.
        title : str
            The title to display on the tab.

        Returns
        -------
        None
        """
        # Ensure label has its natural size so scrollbars appear if needed
        if label.pixmap() is not None:
            img_size = label.pixmap().size()
            label.resize(img_size.width(), img_size.height())
            label.setMinimumSize(img_size.width(), img_size.height())

        # Wrap in a scroll area
        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(False)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Put scroll area inside a container (for future extensibility)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

        # Add to tab widget
        self.tabWidget.addTab(container, title)


    def load_images(self, update_images=None):
        """
        Load images into tabs from the input folder if no update_images are given,
        or update the existing tabs' images if update_images are provided.

        Parameters
        ----------
        update_images : list, optional
            Image data to update existing tabs. If None, images are loaded
            from the input folder according to the selected measurement type.

        Returns
        -------
        None.
        """

        # Case 1: Initial load from folder (no update_images provided)
        if update_images is None:
            # Determine measurement type and input path
            measType = ps.Meas_Type.from_window(self)
            text = self.input_folder()
            path = Path(text)

            if not path.is_dir():
                QMessageBox(QMessageBox.Warning,
                            "Warning",
                            f"Image folder not found:\n'{path}'\n\n"
                            f"Use Select Folder to pick the image folder.",
                            parent=self).exec()
                return

            # Resolve image paths via your settings (already sorted)
            image_paths = measType.settings.img_from_path(path)

            if image_paths:
                image_paths = self.confirm_image_order(path, image_paths, measType)
                if not image_paths:
                    # User cancelled the ordering dialog: keep current tabs.
                    return

            # Clear existing tabs safely
            while self.tabWidget.count():
                w = self.tabWidget.widget(0)
                if w is not None:
                    w.deleteLater()
                self.tabWidget.removeTab(0)

            # Load raw images and keep a copy
            raw_images = [ps.load_image(image_path, measType) for image_path in image_paths]
            self.images = raw_images  # preserve original data as before
            self._loaded_paths = list(image_paths)

            # Convert to QImage for display
            qimages = [self.qimage_fromdata(img) for img in raw_images]

            # Build tabs: MyLabel -> QScrollArea -> Tab
            for ii, qimg in enumerate(qimages):
                # Create your label and set image
                label = MyLabel(mainwindow=self)
                label.set_outline(self.dial.value())
                label.setPixmap(QPixmap.fromImage(qimg))
                self.add_label_as_tab(label, f"img{ii}")

        # Case 2: Update existing tabs with new images
        else:
            # Convert incoming images to QImage
            qimages = [self.qimage_fromdata(img) for img in update_images]

            # Update each existing tab's label (up to the number of tabs and provided images)
            tab_count = self.tabWidget.count()
            n = min(tab_count, len(qimages))
            for i in range(n):
                qimg = qimages[i]
                label = self.get_label_from_tab(i)
                if label is None:
                    continue

                # Update pixmap and resize label to new image size
                label.setScaledContents(False)
                label.setPixmap(QPixmap.fromImage(qimg))

                img_size = qimg.size()
                label.resize(img_size.width(), img_size.height())
                label.setMinimumSize(img_size.width(), img_size.height())

                # If the tab is a QScrollArea, ensure policies remain correct
                tab_content = self.tabWidget.widget(i)
                if isinstance(tab_content, QWidget):
                    # Find a QScrollArea inside container (if layout was used)
                    scroll_areas = tab_content.findChildren(QScrollArea)
                    for scroll in scroll_areas:
                        scroll.setWidgetResizable(False)
                        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)


    def _preview_pixmap(self, image_path, measType=None):
        """Load a single image file as a QPixmap for the order dialog preview."""
        try:
            if measType is not None:
                img = ps.load_image(image_path, measType)
            else:
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return None
            return QPixmap.fromImage(self.qimage_fromdata(img))
        except Exception:
            return None

    def confirm_image_order(self, folder, image_paths, measType):
        """Confirm the time order of images with the user if names are uncoded.

        Returns the ordered list of paths, or an empty list if the user
        cancels. A remembered order for the folder is reused silently.
        """
        predicted, needs_confirm = ps.infer_image_order(image_paths)
        if not needs_confirm:
            return predicted

        folder_key = str(folder)
        current_names = [p.name for p in predicted]
        remembered = self._manual_orders.get(folder_key)
        if remembered is not None and sorted(remembered) == sorted(current_names):
            by_name = {p.name: p for p in image_paths}
            return [by_name[n] for n in remembered if n in by_name]

        by_name = {p.name: p for p in image_paths}
        cache = {}

        def preview_loader(name):
            if name not in cache:
                cache[name] = self._preview_pixmap(by_name[name], measType)
            return cache[name]

        dialog = ReorderDialog(
            self, file_names=current_names, preview_loader=preview_loader
        )
        if dialog.exec_() != QDialog.Accepted:
            return []
        ordered_names = dialog.ordered_names()
        self._manual_orders[folder_key] = list(ordered_names)
        return [by_name[n] for n in ordered_names if n in by_name]

    def reorder_current_images(self):
        """Re-open the ordering dialog for the currently loaded images."""
        if not self.images or not self._loaded_paths:
            QMessageBox.information(
                self, "Reorder Images", "Load images first, then reorder them."
            )
            return
        # Note: extra tabs (e.g. edges/domain_fit) are left untouched; only
        # the loaded base images are reordered.
        names = [p.name for p in self._loaded_paths]
        loaded = {p.name: img for p, img in zip(self._loaded_paths, self.images)}

        def preview_loader(name):
            img = loaded.get(name)
            if img is None:
                return None
            return QPixmap.fromImage(self.qimage_fromdata(img))

        dialog = ReorderDialog(self, file_names=names, preview_loader=preview_loader)
        if dialog.exec_() != QDialog.Accepted:
            return
        ordered_names = dialog.ordered_names()
        by_path = {p.name: p for p in self._loaded_paths}
        self._loaded_paths = [by_path[n] for n in ordered_names if n in by_path]
        self.images = [loaded[n] for n in ordered_names if n in loaded]
        folder_key = self.input_folder()
        self._manual_orders[folder_key] = list(ordered_names)
        # Refresh tab pixmaps in place so drawn direction lines are kept.
        self.load_images(self.images)

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
    
    def show_circle_fit(self):

        binarize = ps.Binarize_Type.from_window(self)
        bin_imgs = binarize.binarize_list(images= self.images)

        measType = ps.Meas_Type.from_window(window = self)
        measType.index = ps.Meas_Type.BUBBLE_CIRCLE_FIT
        measType.select_domain = False
        motions = ps.cexpand_detect(bin_imgs, measType)

        shape = self.images[0].shape
        new_img = ps.image_from_coords(np.empty((0, 2), dtype=np.int32), shape) # Look here
        for motion in motions:
            [ps.image_from_coords(domain.contour.reshape(-1, 2), shape, new_img)
                for domain in motion.domains]
        
        new_img[new_img > 0] = 255
        new_img = new_img.astype(np.uint8)

        new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)
        
        for motion in motions:

            cen_rad = [cv2.minEnclosingCircle(dom.contour) for dom in motion.domains]

            # print(np.diff([x[1] for x in cen_rad])*measType.settings.scale)

            [cv2.circle(new_img, tuple(np.round(center).astype(int)), round(radius), ps.bd.constant.RED,1) 
             for center, radius in  cen_rad]
 
        label = MyLabel(mainwindow=self)
        label.set_outline(self.dial.value())
        qimg = self.qimage_fromdata(new_img)
        label.setPixmap(QPixmap.fromImage(qimg))
        self.add_label_as_tab(label, "domain_fit")

    def dial_changed(self, value):

        # change the label in qlabel
        self.label_dial.setText(f'Width\n({value})')

        # Changes the outline in the line drawn on all the custom labels
        tabs = [self.get_label_from_tab(i) for i in range(self.tabWidget.count()) if self.tabWidget.widget(i)]
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
        
            label: QLabel = self.get_label_from_tab(ind)
            
            pixmap: QPixmap = label.pixmap()

            if pixmap:
                qimage : QImage = pixmap.toImage()
                image : np.ndarray = ps.qimage_to_array(qimage= qimage)
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
        
        tab = self.get_label_from_tab(self.tabWidget.currentIndex())
        tabs = [self.get_label_from_tab(i) for i in range(self.tabWidget.count()) if self.tabWidget.widget(i)]
        
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
        
        tab = self.get_label_from_tab(self.tabWidget.currentIndex())
        tabs = [self.get_label_from_tab(i) for i in range(self.tabWidget.count()) if self.tabWidget.widget(i)]        
       
        # taking the points cordinates form input
        p1 =  QPoint(self.linep1_x.value(), self.linep1_y.value())
        p2 = QPoint(self.linep2_x.value(), self.linep2_y.value())
        
        # checking check box if it has to be set for all tabs
        if not self.checkBox.isChecked():
            
            # if any of the inputs are non zero line should be redrawn
            if p1 or p2:
                
                tab.set_line_points(p1, p2)
                
            # if all the inputs are zero line should be deleted    
            else:
                
                tab.set_line_points(None, None)
        # applying the logic to all tabs if needed.
        else :
            
            if p1 or p2:
                
                [tab.set_line_points(p1, p2) for tab in tabs]
                
            else:
                
                [tab.set_line_points(None, None) for tab in tabs]


    def set_edge(self):
        binarize = ps.Binarize_Type.from_window(self)
        measType = ps.Meas_Type.from_window(window = self)
        new_img = ps.get_edge(self.images, binarize, measType)
        # new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
        
        label = MyLabel(mainwindow=self)
        label.set_outline(self.dial.value())
        qimg = self.qimage_fromdata(new_img)
        label.setPixmap(QPixmap.fromImage(qimg))
        self.add_label_as_tab(label, "edges")

    def calculate_current(self):
        # Calculating distance
        displacements = ps.calculate_motion_displace(
            self.images,
            ps.Meas_Type.from_window(self),
            ps.Binarize_Type.from_window(self)
        )
        displacements = [x for x in displacements if x]
        displacements = np.array([x for y in displacements for x in y])
        displacement = np.nanmean(displacements)
        std = np.nanstd(displacements)

        # # Format the output
        result_text = f"Displacement: {displacement:.2f}\nStandard Deviation: {std:.2f}"
        

        # # Print into QTextEdit
        self.textOutput.append(result_text)   # appends new line each time
        self.textOutput.append("-" * 40)

        # # or use setPlainText(result_text) if you want to overwrite instead
        # # self.textOutput.append("<hr>") # horizontal rule in HTML  

    @staticmethod
    def _folder_pulse_problem(folder: Path, img_type) -> Union[str, None]:
        """Return a problem description if a folder is unfit for batch processing."""
        try:
            files = [p for p in folder.iterdir()
                     if p.is_file() and np.any([p.name.endswith(x) for x in img_type])]
        except OSError as exc:
            return f"{folder.name}: cannot list folder ({exc})"
        if not files:
            return f"{folder.name}: no images found"
        for f in files:
            stem = re.sub(r'\.\w+$', '', f.name)
            try:
                ps.parse_image_stem(stem)
            except ValueError:
                return f"{folder.name}: '{f.name}' has no coded pulse width"
        return None

    def check_batch_folders(self, paths: Path, volts: List[str], measType) -> List[str]:
        """Scan batch folders in a thread pool for missing pulse-width coding."""
        img_type = measType.settings.get_img_type()
        targets: List[Path] = []
        for volt in volts:
            targets.extend(
                [d for d in paths.iterdir()
                 if d.is_dir() and re.match(rf'^{volt}\D+', d.name)]
            )
        problems: List[str] = []
        if not targets:
            return problems
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(8, len(targets))) as pool:
            for problem in pool.map(
                    lambda folder: MainWindow._folder_pulse_problem(folder, img_type),
                    targets):
                if problem:
                    problems.append(problem)
        return sorted(problems)

    def warn_uncoded_batch_folders(self, problems: List[str]) -> None:
        shown = "\n".join(problems[:10])
        if len(problems) > 10:
            shown += f"\n... and {len(problems) - 10} more"
        QMessageBox(
            QMessageBox.Warning,
            "Batch processing needs coded pulse widths",
            f"These folders cannot be batch-processed because their image "
            f"names do not code the pulse width:\n\n{shown}\n\n"
            f"For batch processing every image name must code the pulse "
            f"width (e.g. 0p44s.png, 0p44s_001.png, 0p44s10.png).\n"
            f"See the image file naming documentation:\n{DOCS_URL}\n\n"
            f"Measure these folders manually instead: load each folder and "
            f"use Calculate Current.",
            parent=self).exec()

    def calculate_all(self):

        folder = Path(self.input_folder())
        if not folder.is_dir():
            QMessageBox(QMessageBox.Warning,
                        "Warning",
                        f"Image folder not found:\n'{folder}'\n\n"
                        f"Check the folder path box.",
                        parent=self).exec()
            return None
        paths = folder.parent
        # out_path = paths/"DisplacementTime"
        out_path = paths/f"{self.result_folder_name()}"

        if not out_path.exists():
            out_path.mkdir()
        else:
            # if the overwrite checkbok is not checked it will skip this directory
            if not self.bulk_overwrite_check.isChecked():
                QMessageBox(QMessageBox.Warning,
                            "Warning",
                            f"Specified Folder '{self.result_folder_name()}' Already Exists in the directory'{paths.name}'",
                            parent= self).exec()
                return None

        volts= ps.find_volt(paths)

        arguments = dict(
                        measType = ps.Meas_Type.from_window(window = self),
                        binarize = ps.Binarize_Type.from_window(window = self)
                        )

        problems = self.check_batch_folders(paths, volts, arguments['measType'])
        if problems:
            self.warn_uncoded_batch_folders(problems)
            return None


        do_partial =  partial(do, paths = paths, out_path = out_path, **arguments)    
        # with Pool(4) as p:

        #     results = p.map(do_partial, volts)    
        pool = self.pool.start()
        result = pool.map_async(do_partial, volts)

        
        thread = Thread(target= self.manage_calculate_all, args=(result,))
        thread.start()

        # for x in volts:
        
        #     out = ps.float_str(ps.field(float(x)/10),2) + 'mT.txt'
            
        #     dat = self.dt_curve(paths, x)
            
        #     dat.to_csv(out_path/out, sep = '\t', index = False)
    
    def manage_calculate_all(self, result):

        self.calcAllButton.setText("Stop")
        self.bulkcalcButton.setEnabled(False)
        try:
            while self.pool.pool is not None:
                if result.ready():
                    self.pool.wait()
                    break
                result.wait(timeout=1)
            
        except Exception as e:
            print(e)
            pass
        finally:
            # Keep Bulk Calculate disabled (enable here when the feature is wanted).
            self.bulkcalcButton.setEnabled(False)
            self.calcAllButton.setText("Calculate All")

    def stop_calculate_all(self):

        self.pool.terminate()
            
    def calculate_all_from_parent(self):

        folder = Path(self.input_folder())
        if not folder.is_dir():
            QMessageBox(QMessageBox.Warning,
                        "Warning",
                        f"Image folder not found:\n'{folder}'\n\n"
                        f"Check the folder path box.",
                        parent=self).exec()
            return None
        paths_parent = folder.parent.parent

        folder_name = self.result_folder_name()
        

        
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

                problems = self.check_batch_folders(paths, volts, arguments['measType'])
                if problems:
                    self.warn_uncoded_batch_folders(problems)
                    continue

                do_partial =  partial(do,paths = paths, out_path = out_path, **arguments)
                
                with Pool(4) as p:

                    results = p.map(do_partial, volts)    
                    
                    
    def plot(self, path: Union[str, Path]):
        '''
        Plot the data in the current save folder
        Parameters
        ----------
        path : str|Path
            path of the dir where the files are.

        Returns
        -------
        None.

        '''
        # pattern = re.compile(r"\d+\.\d+mT\.txt")
        pattern = re.compile(r'^[+-]?\d+(\.\d+)?\D+')


        path = Path(str(path).strip())

        if not path.is_dir():
            QMessageBox(QMessageBox.Warning,
                        "Warning",
                        f"Result folder not found:\n'{path}'\n\n"
                        f"Run Calculate All first, then plot.",
                        parent=self).exec()
            return

        files = [x for x in path.iterdir() if pattern.match(x.name)]

        if not files:
            QMessageBox(QMessageBox.Warning,
                        "Warning",
                        f"No result files found in:\n'{path}'",
                        parent=self).exec()
            return
        
        fig, ax = plt.subplots()
        # sorting the files according to the field value
        files.sort(key = lambda x: float(re.match(r'^[+-]?\d+(\.\d+)?', x.name).group()))
        
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

    def open_img_viewer(self):
        # self.img_viewer_open = True
        if self.img_viewer is not None:

            if self.img_viewer.isVisible():
                self.img_viewer.show()
                self.img_viewer.activateWindow()
            else:
                self.img_viewer = Modvortex_ImageProcessor(parent= self,
                                                        worker= Worker)
                self.img_viewer.show()
        else:

            self.img_viewer = Modvortex_ImageProcessor(parent= self,
                                                    worker= Worker)
            self.img_viewer.show()
    

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
        if self.img_viewer is not None:
            if self.img_viewer.isVisible():
                self.img_viewer.close()
            else:
                self.img_viewer = None

        self.pool.terminate()
        self.stop_api()
        event.accept()

    # def start_api(self):
    #     self.flask_thread = Thread(target=self.flask_app.app.run, kwargs={'debug': True, 'use_reloader': False})
    #     self.flask_thread.start()
    #     self.api_running = True
    #     self.api_button.setText('Stop API')

    # def stop_api(self):
    #     if self.api_running:
    #         requests.post('http://127.0.0.1:5000/shutdown', json={'secret_key': self.secret_key})
    #         self.flask_thread.join()
    #         self.api_running = False
    #         self.api_button.setText('Start API')


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
            
            if self.outline > 2:
    
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
        self.img_type : int

        self.key_list = ['scale', 'kernel', 'img_width', 'img_height', 'img_type'] # these are the attributes of this subclass

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
        self.img_type_box.currentIndexChanged.connect(self.check_changes)
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
            data['img_type'] = int(data['img_type'])
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
        self.img_type_box.setCurrentIndex(self.img_type)

    def set_value(self) -> None:
        """
        Set the instance variables from the widget values.
        """
        self.scale = self.scale_box.value()
        self.kernel = self.gaussian_box.value()
        self.img_width = self.img_width_box.value()
        self.img_height = self.img_height_box.value()
        self.img_type = self.img_type_box.currentIndex()
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
                self.img_type_box.setCurrentIndex(int(data['img_type']))
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
                'img_height': self.img_height_box.value(),
                'img_type' : self.img_type_box.currentIndex()
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
            self.img_height_box.value() != self.img_height or
            self.img_type_box.currentIndex() != self.img_type):
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