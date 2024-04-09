# -*- coding: utf-8 -*-
"""
Created on Fri May 12 00:20:26 2023

@author: Rakhul Raj
"""

import cv2
import pathlib
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
import processing as ps
Point =  namedtuple('Point', ['x', 'y'])

img1 = cv2.imread(r'\\USER-PC\Data\Rakhul\Domain Motion\DMI\H2933\16.92_mT\6.0V_3\1p4s_001.png', 0)[:512] # 0 means grayscale
img2 = cv2.imread(r'\\USER-PC\Data\Rakhul\Domain Motion\DMI\H2933\16.92_mT\6.0V_3\1p4s.png', 0)[:512]

img = pathlib.Path(r'\\USER-PC\Data\Rakhul\Domain Motion\DMI\H2933\16.92_mT\6.0V_3').glob('*.png')

img = [cv2.imread(str(x), 0)[:512] for x in img]

def contour_center(contour):
    ''' find the center of a contour'''
    ''' find the center of a contour'''
    
    M = cv2.moments(contour)
    if M['m00'] == 0:
        cX = 0
        cY = 0
    else:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
    center = Point(cX, cY)
    return center


def distance(point1, point2):
    
    p1 = point1; p2 = point2
    
    return np.sqrt((p2.x- p1.x)**2 + (p2.y - p1.y)**2)

def binarize(image):
    img_gus = cv2.GaussianBlur(image, (25,25),0)
    
    # th, ret = cv2.threshold(img_gus, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    th, ret = cv2.threshold(img_gus, 104, 255, cv2.THRESH_BINARY)
    
    
    return ret

def get_contours(image):
    '''take a binarized image as input and outputs the contours'''
    
    contours, hei = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return contours

# c1 = max(contours, key=lambda x : cv2.contourArea(x))
def select_domain(image):
    contours = get_contours(image)
    c1 = min(contours, key=lambda x : distance(contour_center(x), Point(255,255)))
    
    return c1

c1 = select_domain(binarize(img2))

img = [binarize(x) for x in img]

dws = [select_domain(image).squeeze() for image in img]

dom_list = ps.Domain_mot(dws)

dom_list.get_intersect((ps.Point(342, 512), ps.Point(342,1)))

def mark_center(contour, image, color = None, radius = 3):
    '''
    Mark center of a contour in the provided image

    Parameters
    ----------
    contour : TYPE
        The contour.
    image : TYPE
        The image to be drawn in.
    color : TYPE, optional
        DESCRIPTION. The default is None.
    radius : TYPE, optional
        radius of the dot. The default is 3.

    Returns
    -------
    None.

    '''
    # Calculate the center of the selected contour
    center = contour_center(contour)

    # drawing circle around the contour center
    cv2.circle(image, (center.x, center.y), 5, 255, -1)

# Draw all the contours
cv2.drawContours(img2, [c1], -1, 225, 3)
mark_center(c1, img2)
# cv2.contourArea(cnt)
# Show the img2
cv2.imshow("Contours",img2)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

plot_c = c1.reshape((-1,2))
plt.plot(plot_c[:,0], plot_c[:,1])

#%%
# for resolving extra data points in the displacement time graph
from pathlib import Path
import processing as ps
paths = Path(r'\\USER-PC\Data\Users_2023\UPES_UK\Anmol\Domain Motion\CFB_9_100C_2nd')
volts = ps.find_volt(paths)
voltage = '0.8'
[path for path in list(paths.glob(f'{voltage}*'))]