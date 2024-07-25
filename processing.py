# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 17:06:11 2023

@author: Rakhul Raj
"""
# from typing import TYPE_CHECKING
import cv2
import sys
from numba import njit
import numpy as np
import pandas as pd
import pathlib
import re
from collections import namedtuple
from shapely.geometry import LineString as lstr
from scipy.interpolate import interp1d
from shapely.geometry import MultiPoint, GeometryCollection, MultiLineString
from shapely.geometry import Point as Pnt
from scipy.spatial import KDTree
from PyQt5.QtCore import Qt, QPoint


# if TYPE_CHECKING:
from typing import Tuple, List, Any, TypeVar
from PyQt5.QtWidgets import QWidget
# import numpy.typing as npt

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



x5 = (500/188)
x20 = (100/149)

file_list: List[pathlib.Path]= []
file_list.append(pathlib.Path(r'E:\Data\Rakhul\Domain Motion\V_Field_PCoil.txt'))
file_list.append(pathlib.Path(r'\\USER-PC\Data\Rakhul\Domain Motion\V_Field_PCoil.txt'))
file_list.append(pathlib.Path(r'D:\Lab Data\Domain Motion\V_Field_PCoil.txt'))

for cal_file in file_list:
    if cal_file.exists():
        break
if not cal_file.exists():
    raise FileNotFoundError('Calibration File Not Found')

cal = pd.read_csv(cal_file, delimiter ='\t', header = None, usecols = [0,1])

# =============================================================================
# funcion for interpoting the magnetic field from voltage value
# =============================================================================
field = interp1d(cal[0], cal[1])

class Meas_Type():
    
    def __init__(self, window):
        
        self.index = window.measurmentType.currentIndex()
        self.center = window.center_select.text()
        if self.center:
            self.center = Point(*map(int, self.center.split(",")))
            
        # point2 is the first click and point1 is second click        
        self.point2 = Point(window.linep1_x.value(), window.linep1_y.value())
        self.point1 = Point(window.linep2_x.value(), window.linep2_y.value())
        self.outline =  window.dial.value()
        
    def __eq__(self, other):
        
        return self.index == other

# cli or code insert type of mesurementType and binarize
class Binarize_Type():
    
    def __init__(self, threshold:int,  inverse:bool, threshold_value:int = 149):
        """
        

        Parameters
        ----------
        threshold : int
            1 for otsu, 0 for custom.
        inverse : bool
            if inversing the image necessary.
        threshold_value : int, optional
            threshold value that should be used for custom. The default is 149.

        Raises
        ------
        ValueError
            if value of threshold not zero or one then raises error.

        Returns
        -------
        Binarize_Type_cli object.

        """
        self.index = threshold
        if threshold == 1:
            self.threshold = 'otsu'
        elif threshold == 0:
            self.threshold = threshold_value
        else:
            raise ValueError("threshold can only be 0 or 1 zero for custom and 1 for otsu")
        
        self.inverse  = inverse
        
    @classmethod
    def from_window(cls, window: QWidget):
        index = window.b_combo_box.currentIndex()
        if index == 1:
            threshold = 'otsu'
        else:
            threshold = window.spinBox.value()
        inverse = window.inverse.isChecked()
        return cls(index, inverse, threshold)
        
    def __eq__(self, other):
        
        return self.index == other
    
    def binarize_list(self, images: List[np.array]):
        """
        This function binarizes a list of input images based on the instance's threshold and inverse attributes.
    
        Parameters:
        images (list of np.array): The input images to be binarized. They should be grayscale images.
    
        Returns:
        bin_imgs (list of np.array): The binarized images.
    
        The function first applies a Gaussian blur to each image. Then, depending on the threshold attribute of the instance,
        it applies Otsu's binarization or simple thresholding. If the inverse attribute is True, the binary images are inverted.
        """
        # Initialize the list to store the binarized images
        bin_imgs = []
        
        # Determine the type of thresholding to be applied based on the inverse attribute
        type_of_self = cv2.THRESH_BINARY_INV if self.inverse else cv2.THRESH_BINARY
        
        # Apply Gaussian blur to each image
        img_guses = [ cv2.GaussianBlur(img, (25,25),0) for img in images ]
        
        # Iterate over each blurred image
        for ii, img_gus in enumerate(img_guses):
            # Check if the threshold attribute is set to 'otsu'
            if self.threshold == 'otsu':
                # If it's the first image
                if ii == 0:
                    # Concatenate the blurred image with the last blurred image along the horizontal axis
                    img_ext = np.concatenate( (img_gus, img_guses[-1]), axis=1, dtype=np.uint8)
                    # Apply Otsu's binarization
                    th, ret = cv2.threshold(img_ext, 0, 255, type_of_self + cv2.THRESH_OTSU)
                    # Split the binarized image and keep the first half
                    ret = np.split(ret, [img_ext.shape[1]//2], axis=1)[0]
                    # Convert the image from grayscale to BGR and back to grayscale
                    ret = cv2.cvtColor(ret, cv2.COLOR_GRAY2BGR)
                    ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
                else : 
                    # Apply Otsu's binarization for the other images
                    th, ret = cv2.threshold(img_gus, 0, 255, type_of_self + cv2.THRESH_OTSU)
            else:
                # Apply simple thresholding if the threshold attribute is not set to 'otsu'
                th, ret = cv2.threshold(img_gus, self.threshold , 255, type_of_self)
            
            
            # yield ret
            # Append the binarized image to the list
            bin_imgs.append(ret)
            
        # Return the list of binarized images
        return bin_imgs
        
    def binarize(self, image: GrayImage) -> GrayImage:
        """
        This function binarizes an input image based on the instance's threshold and inverse attributes.
    
        Parameters:
        image (np.array): The input image to be binarized. It should be a grayscale image.
    
        Returns:
        ret (np.array): The binarized image.
    
        The function first applies a Gaussian blur to the image. Then, depending on the threshold attribute of the instance,
        it applies Otsu's binarization or simple thresholding. If the inverse attribute is True, the binary image is inverted.
        """
        # Determine the type of thresholding to be applied based on the inverse attribute
        type_of_self = cv2.THRESH_BINARY_INV if self.inverse else cv2.THRESH_BINARY
    
        # Apply Gaussian blur to the image
        image_gus = cv2.GaussianBlur(image, (25,25),0)
    
        # Check if the threshold attribute is set to 'otsu'
        if self.threshold == 'otsu':
            
            # Apply Otsu's binarization
            th, ret = cv2.threshold(image_gus, 0, 255, type_of_self + cv2.THRESH_OTSU)
            # # Convert the image from grayscale to BGR and back to grayscale
            # ret = cv2.cvtColor(ret, cv2.COLOR_GRAY2BGR)
            # ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
        else:
            # Apply simple thresholding if the threshold attribute is not set to 'otsu'
            th, ret = cv2.threshold(image_gus, self.threshold , 255, type_of_self)
    
        # Return the binarized image
        return ret

def float_str(number, dec_places):
    
    number = str(np.round(number, dec_places))
    num_list = number.split('.')
    if len(num_list[-1]) < dec_places:
        
        num_list[-1] += (dec_places - len(num_list[-1]))*'0'
    number = '.'.join(num_list)
    return number

def  find_volt(path: pathlib.Path):
    '''
    takes path conataining voltages as input and output a list of voltages in the path

    Parameters
    ----------
    path : pathlib.Path
        path conataining voltages.

    Returns
    -------
    a : list
        list of voltages.

    '''
    a = []
    for x in path.glob('[0-9]*'):
        name = x.name.split('_')[0][:-1]
        if not name in a:
            a.append(name)
    a.sort(key = lambda x : float(x))
    
    return a

def get_edge(image : GrayImage, get_cordinates : bool = False) -> np.array:
    '''
    Take the image and give out image with edges or the coordinates of the edge

    Parameters
    ----------
    image : np.array
        input image.
    get_cordinates : bool, optional
        if true the coordinate are given. The default is False.

    Returns
    -------
    np.array
        image with edges or coordinates.

    '''
    img1 = image
    # getting images with only edges
    edges1: GrayImage = cv2.Canny(img1, 200, 400)
    
    if get_cordinates:
        
        return cv2.findNonZero(edges1)
    else:
        return edges1
    
def dw_select(image : GrayImage ) -> GrayImage:
    '''
    

    Parameters
    ----------
    image : np.array
        DESCRIPTION.

    Returns
    -------
    points : TYPE
        DESCRIPTION.

    '''
    contours1 : Contour ; hierarchy1: np.array
    contours1, hierarchy1 = cv2.findContours(image, 
                                             cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
    if not contours1:
        return None
    c1 = max(contours1, key=lambda x : cv2.arcLength(x, True))
    edges1 = cv2.findNonZero(image)
    # points inside the contour with biggest perimeter
    points = [g for g in edges1.squeeze()
              if cv2.pointPolygonTest(c1,(int(g[0]),int(g[1])),False) >= 0 ]
    
    return points

@njit(cache = True)
def image_from_coords(coords : np.array, 
                      shape : tuple, new_image : np.array = None) -> GrayImage:
    '''
    

    Parameters
    ----------
    coords : np.array
        DESCRIPTION.
    shape : tuple
        DESCRIPTION.
    new_image : np.array, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    new_image : TYPE
        DESCRIPTION.

    '''
    if new_image is None:
        
        new_image  = np.zeros(shape, np.uint8)
        
    # dws_contours = [np.expand_dims(coords, axis=1)] 
    
    # cv2.drawContours(new_image, dws_contours, 0, [255,255,255], 1) # this will not work for random shaped domains
        
    for x in coords:
        
        new_image[x[1],x[0]] = 1
        
    return new_image
@njit(cache = True)
def find_endpoints(edges : np.array, shape : Tuple) -> List[np.array]:
    # not a good way to find endpoints for 
    endpoints = [np.array([x], dtype = 'int32') for x in range(0)] # empty list
    
    ii = 0
    
    while not endpoints:
        
        endpoints = [x for x in edges if x[0] == 0+ii or x[1] == 0+ii or 
                     x[0] == shape[1]-1-ii or x[1] == shape[0]-1-ii]
        ii += 1
        
        if ii > shape[1]-1-ii or ii > shape[0]-1-ii:            
            break
        
    return endpoints


# # new function to be implemented, testing is needed before implementation
# # Function to find endpoints
# def find_endpoints(edges : np.array, shape : Tuple) -> List[Tuple]:

#     image : GrayImage = image_from_coords(edges, shape)
#     image[image == 1] = 255
#     # Skeletonize the edge-detected image
#     skeleton = cv2.ximgproc.thinning(edges)

#     # Find non-zero points in the skeleton
#     points = np.argwhere(skeleton > 0)


#     endpoints = []
#     for point in points:
#         y, x = point
#         neighbors = skeleton[y-1:y+2, x-1:x+2]
#         if np.sum(neighbors) == 2:  # Only one neighbor
#             endpoints.append((x, y))
#     return endpoints
    
def ordered_edge(edges : np.array, shape : Tuple) -> np.array:
    '''
    

    Parameters
    ----------
    edges : np.array
        DESCRIPTION.
    shape : tuple
        DESCRIPTION.

    Returns
    -------
    ordered_edges : np.array
        DESCRIPTION.

    '''

    endpoints = find_endpoints(edges, shape)

    point  = endpoints[0]
    
    # Create a KDTree from the coordinates
    tree = KDTree(edges)
    
    test = [ x for x in endpoints if point[0] == x[0]]
    test_1 = [x for x in endpoints if point[1] == x[1]]
    
    if len(test) > 1:
        
        indexes = [len(tree.query_ball_point(point, np.sqrt(2))) for point in test]
        start = test[indexes.index(min(indexes))]
            
    elif len(test_1) > 1:
        
        indexes = [len(tree.query_ball_point(point, np.sqrt(2))) for point in test_1]
        start = test_1[indexes.index(min(indexes))]
        
    else :
        
        start = endpoints[0]        
        
    edges_copy = edges.copy()

    # Initialize a list to store the ordered coordinates
    ordered_edges = []
    # edges.remove(start)
    # Loop until all points are visited
    while len(ordered_edges) < len(edges_copy):
        tree = KDTree(edges)
        # Find the nearest neighbor of the current point
        if ordered_edges:
            dist, index = tree.query(ordered_edges[-1], k=1)
        else:
            dist, index = tree.query(start, k=1)
        # Get the coordinates of that neighbor
        next_point = edges[index]
        # Append it to the ordered list
        ordered_edges.append(next_point)
        # Remove it from the original list (to avoid visiting it again)
        edges = np.delete(edges, index, axis=0)
        # Update the current point
        start = next_point
    ordered_edges =  np.array(ordered_edges)   
        
    return ordered_edges

def dw_detect(image: GrayImage):
    
    edges1 : GrayImage = get_edge(image)
    dw1 = dw_select(edges1)
    if dw1:
        ordered_dw = ordered_edge(dw1, image.shape)
        
        return ordered_dw


class Domain_mot():
    
    def __init__(self, walls, scale  = x20):
        
        self.walls = walls
        self.scale = scale
        
    def get_intersect(self, line):
        
        dws = self.walls
        asline = [lstr(x) for x in dws]
        # point1 = QPoint(*line[0])
        # point2 = QPoint(*line[1])
        point1 = (line[0].x, line[0].y)
        point2 = (line[1].x, line[1].y)
        intersect = [line.intersection(lstr([point1, point2])) for line in asline]
    
        return intersect
    
    # distance = [np.sqrt((x.x - y.x)**2 + (x.y-y.y)**2 ) for x, y in zip(intersect[:-1], intersect[1:])]
    
    def distance(self, line):
        
        intersect = self.get_intersect(line)  
        distance  = []
        # intersect= [list(x.geoms)[0] if isinstance(x, MultiPoint) else x  for x in intersect]
        intersect= [list(x.geoms)[0] if isinstance(x, GeometryCollection) else x  for x in intersect]
        intersect= [list(x.geoms)[0] if isinstance(x, MultiLineString) else x  for x in intersect]
        intersect= [list(x.geoms)[0] if isinstance(x, MultiPoint) else Pnt(x.coords[0]) if isinstance(x, lstr) and x  else x  
                    for x in intersect ]
        # intersect= [list(x.geoms)[0] if not isinstance(x, Pnt) else x  for x in intersect]
        i = 0
        b = False
        for x, y in zip(intersect[:-1], intersect[1:]):
            
            if x and y:
                distance.append(np.sqrt((x.x - y.x)**2 + (x.y-y.y)**2 ) * self.scale)
                b = True
            elif i==0 or not b:
                continue
            else:
                break
            i=+1
        return distance
    
def parallel_lines(point2, point1, outline):
    
    
    x2 = point2.x
    y2 = point2.y
    x1 = point1.x
    y1 = point1.y
    
    lines = []
    for out in range(0, outline, 2):
        
        if (x2 - x1) == 0:
          
            xx = out
            yy = 0 
        
        else:
            
            m = (y2-y1) / (x2-x1)
            xx = int(np.round(m/np.sqrt(1+m**2) * out))
            yy = int(np.round(-1/np.sqrt(1+m**2) * out))
        
        new_first_click = Point( x2+xx, y2+yy )
        new_second_click =  Point( x1+xx, y1+yy )
        
        lines.append((new_first_click, new_second_click))
        
        if out>0:
            
            xx = -xx
            yy = -yy
            
            new_first_click = Point( x2+xx, y2+yy )
            new_second_click =  Point( x1+xx, y1+yy )
            lines.append((new_first_click, new_second_click))
            
    return lines


def get_pwidth(path: pathlib.Path) -> Tuple[float, str]:
    """
    extract the pulse width and unit from the domain image file name

    Parameters
    ----------
    path : pathlib.Path object
        Input the path of the image file.

    Returns
    -------
    width : float
        The pulse width.
    unit : str
        The unit of pulse width.

    """

    name = path.name.replace('.png','').split('_')[0]

    unit = re.findall(r'[a-z]+$', name)[-1]
    a = re.split(r'[a-z]+$', name)[0].split('p')
    width = float('.'.join(a))

    return width, unit


# part needed for dmi measurements


Point =  namedtuple('Point', ['x', 'y'])


def contour_center(contour):
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

def distance(point1 : Point, point2 : Point) -> float:
    '''
    Find the distance between two points

    Parameters
    ----------
    point1 : Point
        1st point.
    point2 : Point
        2nd point.

    Returns
    -------
    float
        the distance.

    '''
    
    p1 = point1; p2 = point2
    
    return np.sqrt((p2.x- p1.x)**2 + (p2.y - p1.y)**2)

def get_contours(image):
    '''take a binarized image as input and outputs the contours'''
    
    contours, hei = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return contours


def bdw_detect(image, center):
    contours = get_contours(image)
    # contours = filter(lambda x : cv2.contourArea(x) > 1, contours)
    # c1 = min(contours, key= lambda x : distance(contour_center(x), Point(255,255)))
    try :
        
        c1 = min(contours, key= lambda x : distance(contour_center(x), center))
    except ValueError as err:
        
        c1 = np.array([])
    
    return c1.squeeze()
#1 implementation remaining - color and radius
def mark_center(contour, image, color = None, radius = 3):
    '''
    Mark center of a contour in the provided image

    Parameters
    ----------
    contour :
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

# def dt_curve(paths : pathlib.Path, voltage : str, measurmentType, binarize) -> pd.DataFrame:
#     '''
#     Takes the parent path which contains the voltage measurements and returns 
#     a displacement v/s pulse_width data
    
#     Parameters
#     ----------
#     path : pathlib.Path
#         Parent path which contain the voltage folders.
#     voltage : str or int
#         Starting of the voltage folder names. For example if the folders are 25V,25V_1
#         25V_2, 25V_3 etc then 25 or 25V can be given as voltage.
    
#     Returns
#     -------
#     dta1 : Pandas DataFrame
#         Returns displacement v/s pulse_width data.
    
#     '''
        
#     dta = pd.DataFrame(columns = ['pulse_width', 'displacement'])
    
#     for path in list(paths.glob(f'{voltage}V*')):
#         # print(path)
#         images = [*path.glob('*.png')]
#         pulse_width = get_pwidth(images[0])[0]
#         images = [cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)[:512] for image in images]
#         displacement = calculate_motion(images, measurmentType, binarize)
#         displacement = [x for x in displacement if x]
#         displacement = np.mean([np.mean(dis) for dis in displacement])
        
#         dta.loc[len(dta)] = [pulse_width, displacement]
#     dta1 =  pd.DataFrame(columns = ['pulse_width', 'displacement'])
#     dta['pulse_width'] = np.round(dta['pulse_width'],6)
#     dta =  dta[np.logical_not(np.isnan(dta['displacement']))]
#     for x in set(dta['pulse_width']):
#         if not np.isnan( x ):
            
#             #dataframe.append will be deprecated in future pandas so going to us concat insted
#             #dta1 = dta1.append(dta[dta['pulse_width']==x].mean(),ignore_index=True)
#             dta1 = pd.concat([dta1,dta[dta['pulse_width']==x].mean().to_frame().T],ignore_index=True)
#     dta1.sort_values(by = 'pulse_width', inplace = True)
#     dta1.reset_index(drop = True, inplace = True)
#     return dta1

def dt_curve_first(paths : pathlib.Path, pulse_select : int,  voltage : str, measurmentType: Meas_Type, binarize: Binarize_Type) -> pd.DataFrame:
    '''
    Takes the parent path which contains the voltage measurements and returns 
    a displacement v/s pulse_width data. it only considers the first pulse for the displacement measurement
    
    Parameters
    ----------
    path : pathlib.Path
        Parent path which contain the voltage folders.
        
    pulse_select : int
        which pulse should used for 
    voltage : str or int (recommends str)
        Starting of the voltage folder names. For example if the folders are 25V,25V_1
        25V_2, 25V_3 etc then 25 or 25V can be given as voltage.
    
    Returns
    -------
    dta1 : Pandas DataFrame
        Returns displacement v/s pulse_width data.
    
    '''
        
    dta = pd.DataFrame(columns = ['pulse_width', 'displacement', 'std'])
    data = {}
    for path in list(paths.glob(f'{voltage}V*')):
        # print(path)
        images = [*path.glob('*.png')]
        pulse_width = get_pwidth(images[0])[0]
        images = [cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)[:512] for image in images]
        displacement = calculate_motion(images, measurmentType, binarize)
        displacement = [x[0:1] for x in displacement if x]    # [0] here is for selecting the displacement from 1st intercetion
        displacement = np.array([x for y in displacement for x in y])
        
        pulse_width = np.round(pulse_width, 6)
        if not pulse_width in data:
        
            data[pulse_width] = displacement
        
        else:
        
            data[pulse_width] = np.concatenate(( data[pulse_width], displacement))
        
    for pulse_width, displacement in data.items():
    
        if displacement.size > 0:
    
            disp = displacement.mean()
            std = displacement.std()
            dta.loc[len(dta)] = [pulse_width, disp, std]
    
    dta.sort_values(by = 'pulse_width', inplace = True)
    dta.reset_index(drop = True, inplace = True)
    return dta


def dt_curve(paths : pathlib.Path, voltage : str, measurmentType: Meas_Type, binarize: Binarize_Type) -> pd.DataFrame:
    '''
    Takes the parent path which contains the voltage measurements and returns 
    a displacement v/s pulse_width data
    
    Parameters
    ----------
    path : pathlib.Path
        Parent path which contain the voltage folders.
    voltage : str or int (recommends str)
        Starting of the voltage folder names. For example if the folders are 25V,25V_1
        25V_2, 25V_3 etc then 25 or 25V can be given as voltage.
    
    Returns
    -------
    dta1 : Pandas DataFrame
        Returns displacement v/s pulse_width data.
    
    '''
        
    dta = pd.DataFrame(columns = ['pulse_width', 'displacement', 'std'])
    data = {}
    for path in list(paths.glob(f'{voltage}V*')):
        # print(path)
        images = [*path.glob('*.png')]
        pulse_width = get_pwidth(images[0])[0]
        images = [cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)[:512] for image in images]
        displacement = calculate_motion(images, measurmentType, binarize)
        displacement = [x for x in displacement if x]
        displacement = np.array([x for y in displacement for x in y])
        
        pulse_width = np.round(pulse_width, 6)
        if not pulse_width in data:
        
            data[pulse_width] = displacement
        
        else:
        
            data[pulse_width] = np.concatenate(( data[pulse_width], displacement))
        
    for pulse_width, displacement in data.items():
    
        if displacement.size > 0:
    
            disp = displacement.mean()
            std = displacement.std()
            dta.loc[len(dta)] = [pulse_width, disp, std]
    
    dta.sort_values(by = 'pulse_width', inplace = True)
    dta.reset_index(drop = True, inplace = True)
    return dta
# @profile
def calculate_motion(images, measurmentType: Meas_Type, binarize: Binarize_Type):
    
    point2 = measurmentType.point2
    point1 = measurmentType.point1
    outline = measurmentType.outline

    # bin_imgs = [self.binarize(image)[0] for image in images]
    
    bin_imgs = binarize.binarize_list(images= images)
   
    # selecting according to the type of measurements
    if measurmentType == 0:
        
        # if bubble domain type detect the closed contour corresponding to the domain using ps.bdw_detect
        dws = [bdw_detect(image, measurmentType.center) for image in bin_imgs]
        
    elif measurmentType == 1:
        
        
        # if domain of random shape then it probably has a psedo open contour so using ps.dw_detect which
        # uses a edege detection
        dws = [dw_detect(image) for image in bin_imgs]
    # to change scale for auto measurements change here 
    motion = Domain_mot(dws, scale=x20)
    
    lines = parallel_lines(point2, point1, outline)
    
    distance = [motion.distance(line) for line in lines]

    # distance = motion.distance([point2, point1,])

    return distance

# return the domain motion insted of the average displacement
def calculate_domain_motion(images, measurmentType: Meas_Type, binarize: Binarize_Type):
    
    point2 = measurmentType.point2
    point1 = measurmentType.point1
    outline = measurmentType.outline

    # bin_imgs = [self.binarize(image)[0] for image in images]
    
    bin_imgs = binarize.binarize_list(images= images)

    # selecting according to the type of measurements
    if measurmentType == 0:
        
        # if bubble domain type detect the closed contour corresponding to the domain using ps.bdw_detect
        dws = [bdw_detect(image, measurmentType.center) for image in bin_imgs]
        
    elif measurmentType == 1:
        
        
        # if domain of random shape then it probably has a psedo open contour so using ps.dw_detect which
        # uses a edege detection
        dws = [dw_detect(image) for image in bin_imgs]
    # to change scale for auto measurements change here 
    motion = Domain_mot(dws, scale=x20)
    
    return motion


    
class Meas_Type_cli():
    
    def __init__(self, mtype:int, point2:Point, point1:Point, outline:int, 
                 center:Point = Point(371,278)):
        """
        

        Parameters
        ----------
        mtype : int
            0 for closed domain, 1 for random shape.
        point2 : ps.Point
            first click point of the straight line.
        point1 : ps.Point
            2nd click point of the straight line.
        outline : int
            spread of the line, 3.
        center : ps.Point, optional
            DESCRIPTION. The default is ps.Point(371,278).

        Returns
        -------
        None.

        """
        
        self.index = mtype
        # point2 is the first click and point1 is second click        
        self.point2 = point2
        self.point1 = point1
        self.outline =  outline
        self.center = center
        
    def __eq__(self, other):
        
        return self.index == other
    