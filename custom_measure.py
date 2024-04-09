# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:42:00 2023

@author: Rakhul Raj
"""
from pathlib import Path
import processing as  ps
import cv2


paths = Path(r'\\user-pc\Data\Rakhul\Domain Motion\DMI\Repeat\H2933_H1\-1.91_mT')
             
voltage = '5.0'

out_path = None

point2 = ps.Point(350, 273)

point1 = ps.Point(350, 506)

outline = 3 

measurmentType = 0

binarize = 'otsu'

dt = ps.dt_curve(paths, voltage, out_path, point2, point1, outline, measurmentType, binarize)

#%%
path = Path(r'\\user-pc\Data\Rakhul\Domain Motion\DMI\Repeat\H2933_H1\-1.91_mT\5.0V_10')
images = [*path.glob('*.png')]
pulse_width = ps.get_pwidth(images[0])[0]
images = [cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)[:512] for image in images]
displacement = ps.calculate_motion(images, out_path, point2, point1, outline, measurmentType, binarize)
print(displacement)