# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 07:42:37 2023

@author: Rakhul Raj
"""

# Import libraries
import cv2
import numpy as np
import pathlib
from shapely.geometry import LineString as lstr
from shapely.geometry import MultiPoint
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QPoint

x5 = (500/188)
x20 = (100/149)


# Load binarized images
img1 = cv2.imread(r'\\user-pc\Data\Rakhul\Domain Motion\H3186_BTO\296K\4.6V_59\0p015s_001.png', 0)[:512] # 0 means grayscale
img2 = cv2.imread(r'\\user-pc\Data\Rakhul\Domain Motion\H3186_BTO\296K\4.6V_59\0p015s_002.png', 0)[:512]

# img = pathlib.Path(r'\\user-pc\Data\Rakhul\Domain Motion\H3186_BTO\296K\4.6V_59').glob('*.png')
img = pathlib.Path(r'\\user-pc\Data\Rakhul\Domain Motion\H3186_BTO\296K_1\5.4V_6').glob('*.png')

img = [cv2.imread(str(x), 0)[:512] for x in img]

def get_edge(image : np.array, get_cordinates : bool = False) -> np.array:
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
    img_gus = cv2.GaussianBlur(image, (25,25),0)
    # th, img1 = cv2.threshold(img_gus, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th, img1 = cv2.threshold(img_gus, 152, 255, cv2.THRESH_BINARY)
    
    # getting images with only edges
    edges1 = cv2.Canny(img1, 200, 400)
    
    if get_cordinates:
        
        return cv2.findNonZero(edges1)
    else:
        return edges1
    
def dw_select(image : np.array ) -> np.array:
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

def image_from_coords(coords : np.array, 
                      shape : tuple, new_image : np.array = None) -> np.array:
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
        
        new_image  = np.zeros(shape)
        
    for x in coords:
        
        new_image[x[1],x[0]] = 1
        
    return new_image

def find_endpoints(edges : np.array, shape : tuple):
    
    endpoints = []
    
    ii = 0
    
    while not endpoints:
        
        endpoints = [x for x in edges if x[0] == 0+ii or x[1] == 0+ii or 
                     x[0] == shape[1]-1-ii or x[1] == shape[0]-1-ii]
        ii += 1
        
        if ii > shape[1]-1-ii or ii > shape[0]-1-ii:            
            break
        
    return endpoints
    
def ordered_edge(edges : np.array, shape : tuple) -> np.array:
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

def dw_detect(image):
    
    edges1 = get_edge(image)
    dw1 = dw_select(edges1)
    
    if dw1:
        
        ordered_dw = ordered_edge(dw1, image.shape)
    
        return ordered_dw

# new_img = image_from_coords(dw1, edges1.shape)
# image_from_coords(dw2, edges1.shape, new_img)

dws = [(dw_detect(image), print(x))for x,  image in enumerate(img)]
dws = [x[0] for x in dws if not x[0] is None]
new_img = image_from_coords(dws[0], img[0].shape)
[image_from_coords(dw, img[0].shape, new_img) for dw in dws]

# im = img[2]
# dw = dw_select(get_edge(im))
# dw =  ordered_edge(dw, img[0].shape)
# new_img = image_from_coords(dw,img[0].shape)

line = [[506,51], [179,414]]

class Domain_mot():
    
    def __init__(self, walls, scale  = x20):
        
        self.walls = walls
        self.scale = scale
        
    def get_intersect(self, line):
        dws = self.walls
        asline = [lstr(x) for x in dws]
        point1 = QPoint(*line[0])
        point2 = QPoint(*line[1])
        intersect = [line.intersection(lstr([point1.toTuple(), point2.toTuple()])) for line in asline]
    
        return intersect
    
    # distance = [np.sqrt((x.x - y.x)**2 + (x.y-y.y)**2 ) for x, y in zip(intersect[:-1], intersect[1:])]
    
    def distance(self, line):
        intersect = self.get_intersect(line)  
        distance  = []
        intersect= [list(x)[0] if isinstance(x, MultiPoint) else x  for x in intersect]
        
        for x, y in zip(intersect[:-1], intersect[1:]):
            
            if x and y:
                distance.appaend(np.sqrt((x.x - y.x)**2 + (x.y-y.y)**2 ))
            else:
                break
        
        return distance

# # Plot the ordered coordinates as a line plot
# plt.figure(figsize=(12,8))
# plt.plot(ordered_dw1[:,0], -ordered_dw1[:,1], '.-')
# plt.title('Ordered Edge Coordinates')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.savefig("test.svg")
# plt.show()

# # Draw contours on original images for visualization
# img1_c = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) # convert to color for drawing
# img2_c = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
# cv2.drawContours(img1_c, [c1], -1, (0, 255 ,0), 3) # green contour on image 1
# cv2.drawContours(img2_c, [c2], -1, (0 ,0 ,255), 3) # red contour on image 2

# Show images with contours
cv2.imshow('Image 1', img[0])
# cv2.imshow('Image 2', img2_c)
cv2.imshow('Image 2', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Calculate the distance between the centroids of the contours
# M1 = cv2.moments(c1) # moments of contour 1
# M2 = cv2.moments(c2) # moments of contour 2

# # Centroid coordinates
# cx1 = int(M1['m10'] / M1['m00'])
# cy1 = int(M1['m01'] / M1['m00'])
# cx2 = int(M2['m10'] / M2['m00'])
# cy2 = int(M2['m01'] / M2['m00'])

# # Euclidean distance
# distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

# # Print distance in pixels
# print(f'The distance between the edges is {distance:.3f} pixels.')