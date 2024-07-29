# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 22:06:04 2024

@author: Rakhul Raj
"""
import pathlib
import re
import sys
from typing import Any, List, Tuple, TypeVar, Union
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from pint import UnitRegistry, set_application_registry

# ureg = UnitRegistry()
# # for pickling and unpickiling ureg.Quantity. See pint docs
# set_application_registry(ureg)


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

    
class Constant():

    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (125, 125, 125)
    DARK_GRAY = (50, 50, 50)
    LIGHT_GRAY = (220, 220, 220)

constant = Constant()

class B_Domain():

    def __init__(self, contour: Contour, image: GrayImage, centre: tuple = None,):

        self.contour = contour
        self.image = image
        if not centre:
            # self.centre = tuple(np.round(centroid(contour)).astype(int)) # old way
            cnt , _ = cv2.minEnclosingCircle(contour)
            self.centre  =  tuple(map(int, cnt))
        else:
            self.centre = centre
        self.M = cv2.moments(contour)
        self.arc_length = cv2.arcLength(contour, True)
        self.roundness = self.arc_length**2/(self.M['m00']*4*np.pi)
        self.area = cv2.contourArea(contour)

    def __repr__(self):

        return 'Domain at {}'.format(self.centre)

    def distance(self, domain: 'B_Domain'):

        p1 = np.array(self.centre)
        p2 = np.array(domain.centre)
        return np.sqrt(np.sum(np.square(p1-p2)))


class B_Domain_mot():

    def __init__(self, start_domain : B_Domain, pulse: int = 0, scale: float = 1):

        self.domains : List[B_Domain] = [start_domain]
        self.centre : tuple = start_domain.centre
        self.start_pulse = pulse
        self.pulse = pulse
        # self.pulse_width = []
        self.locked = False
        self.scale = scale#*ureg['micrometer'] # look here

    def __repr__(self):

        # return 'Motion of Domain at {}'.format(tuple(self.centre.astype(int)))
        return 'Motion of Domain at {}'.format(self.centre)

    def motion(self, domain : B_Domain):

        if not self.locked:

            self.domains.append(domain)
            self.pulse +=1
            # self.pulse_width.append(pulse_width)
        else:

            pass

    def distance(self, domain: B_Domain):

        p1 = np.array(self.centre)
        p2 = np.array(domain.centre)
        return np.sqrt(np.sum(np.square(p1-p2)))

    def displacement(self):

        radius = [cv2.minEnclosingCircle(d.contour)[1] for d in self.domains]

        displacement =[y-x  for x, y in zip(radius[:-1], radius[1:])]

        displacement = [self.scale * x for x in displacement]

        return displacement

    def plt_motion(self, path: Union[pathlib.Path , str]= None):

        leng = len(self.domains)
        x,y = round(np.ceil(np.sqrt(leng))), round(np.floor(np.sqrt(leng)))

        if x*y < leng:

            x += 1

        plt.figure(figsize =(5*x, 5*y))
        plt.suptitle(str(self))
        for ii, domain in enumerate(self.domains):
            
            l = ii+1
            img = domain.image
            fit_plt_circle(domain.contour, img, copy = False)
            plt.subplot(y,x,l)
            plt.imshow(img[:,:,::-1])
            if ii == 0 :
                name = 'After Nucleation Pulse'
            else:
                name = 'After {} Pulse{}'.format(ii, '' if ii ==1 else 's' )
            plt.title(name)

        if path:

            plt.savefig(path)
        else :

            plt.show()
        # plt.close()
    def lock(self):

        self.locked = True

    def unlock(self):

        self.locked = False


def fit_plt_circle(contour: np.ndarray, img: np.ndarray,
                   circle: tuple = None, copy: bool = True) -> np.ndarray:


    if copy:
        img = img.copy()
    if len(img.shape)==2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if not circle:
        circle = cv2.minEnclosingCircle(contour)
    # print(circle)
    cv2.circle(img, np.round(circle[0]).astype(int),\
                np.round(circle[1]).astype(int), constant.RED, 2)

    return img

        
def centroid(contour):

    M = cv2.moments(contour)

    x = M['m10'] / M['m00']
    y = M['m01'] / M['m00']

    return (x,y)

def save_motion(domains: List[B_Domain], motions: List[B_Domain_mot], scale : float):

    pulses = [motion.pulse for motion in motions]
    pulse = max([motion.pulse for motion in motions])
    for x in domains:
        # generating list of distace from all B_Domain_mot objects
        lengs = []
        for y in motions:

            lengs.append((y, y.distance(x)))

        lengs.sort(key = lambda x: x[1])

        # l= len(list(filter(lambda x : x[1]<5, lengs)))
        l= len(list(filter(lambda x : x[1]<20, lengs))) #finding the closest domain_motion to domain
        if  l == 1:

            lengs[0][0].motion(x)

        elif l >1:

            warnings.warn("more than one domain motion corresponding to the domain '{x}'. Assigning motion to the closest")
            lengs[0][0].motion(x)

        elif l==0:

            pulses.append(pulse)
            motions.append(B_Domain_mot(x,pulse+1,scale = scale))
    # ending the motion if no domain has center matching with the B_Domain_mot object
    for ii, motion in enumerate(motions):

        if motion.pulse == pulses[ii]: # no motion is detected

            motion.lock()

def plt_img(path):

    if type(path) != np.ndarray:

        path = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    path = cv2.cvtColor(path, cv2.COLOR_GRAY2RGB)

    plt.imshow(path)


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
def domain_selector(contour, center):
    p1 = np.round(centroid(contour))
    p2 = center

    return np.sqrt(np.sum(np.square(p1-p2)))

def bdomain_select(image : GrayImage, roundness: float = 1.3, 
                   select_domain: Union[List[Tuple[int]], bool] = False) -> List[B_Domain]:
    
    ''' detect domains from binarized doman image'''

    contours, _hei = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = [x for x in contours if cv2.contourArea(x) != 0]

        # to select specific domains
    if select_domain:
        select_domain = [ np.array(x) for x in select_domain]
        contours = list(filter(lambda x : np.any([domain_selector(x, cen) < 50 for cen in select_domain]),
                                contours))
    
    image_area = np.prod(image.shape)

    #filtering domains from contour
    # domains =  list(filter(lambda i : i.roundness < 1.2, (B_Domain(x,path, scale = scale) for x in contour)))
    domains =  list(filter(lambda i : i.roundness < roundness and  i.area<image_area,
                            (B_Domain(x, image) for x in contours if cv2.minEnclosingCircle(x)[1] < 350)))

    return domains


def analyse_images(images: List[GrayImage] , scale: float = 1, roundness: float = 1.3, 
                   select_domain: Union[List[Tuple[int]], bool] = False) -> List[B_Domain_mot]:
    '''
    analyse images to calculate domain motion and returns a list of B_Domanin_mot object

    Parameters
    ----------
    paths : List[GrayImage]
        list of images from which domain displacment should be measured
    plot : TYPE, optional
        if true, will plot the domains. The default is True.
    save_plot : TYPE, optional
        if true plot and save the domains in a folder named result in paths.
        . The default is False.

    Returns
    -------
    motions : list
        retruns a list of B_Domain_mot objects.

    '''
    motions = []
    for ii, img in enumerate(images):

        domains = bdomain_select(img, roundness, select_domain)


        # for the first image
        if ii==0:

            motions = [B_Domain_mot(x,scale = scale) for x in domains]
        # if there are do domains nucleated in the previous images
        elif len(motions) == 0:

            motions = [B_Domain_mot(x,scale = scale) for x in domains]
        # for images after the detection of domain.
        else :
            save_motion(domains, motions, scale)

    return motions


def plt_motion(motions : List[B_Domain_mot], images : GrayImage, 
               paths : pathlib.Path, plot : bool = False, save_plot : bool = False):

    if (plot and save_plot) or save_plot:

        for ii, motion in enumerate(motions):

            if not (paths/'results').exists():

                (paths/'results').mkdir()

            motion.plt_motion(images, paths/'results'/'{}.jpg'.format(ii))
    elif plot:
        for ii, motion in enumerate(motions):

            motion.plt_motion()