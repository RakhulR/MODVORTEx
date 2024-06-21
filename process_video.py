# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:57:40 2024

@author: Rakhul Raj
"""

# this file is created for processing images converted from video


import processing as ps
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

binarize = ps.Binarize_Type(0, True, 208)

measurmentType = ps.Meas_Type_cli(0, ps.Point(330,262), ps.Point(555,66), 
                 outline = 40, center = ps.Point(330,262) )

measurmentType = ps.Meas_Type_cli(0, ps.Point(327,265), ps.Point(529,52), 
                 outline = 12, center = ps.Point(327,265) )

# measurmentType = ps.Meas_Type_cli(0, ps.Point(327,265), ps.Point(85, 139), 
#                  outline = 12, center = ps.Point(327,265) )

# measurmentType = ps.Meas_Type_cli(0, ps.Point(327,265), ps.Point(429, 429), 
#                  outline = 12, center = ps.Point(327,265) )

# measurmentType = ps.Meas_Type_cli(0, ps.Point(327,265), ps.Point(424, 429), 
#                  outline = 12, center = ps.Point(327,265) )

measurmentType = ps.Meas_Type_cli(0, ps.Point(327,265), ps.Point(299, 430), 
                  outline = 6, center = ps.Point(327,265) )

paths = sorted(Path(r'C:\lab_data_to_update\Useful Programs\output_frames_2p4')\
              .glob('*.png'), key = lambda x: int(x.name.split('.')[0].split('_')[-1]))

paths_name= [*map(lambda x: int(x.name.split('.')[0].split('_')[-1]), paths)]

def get_images(paths):
    for image in paths:
        
        yield ps.cv2.imread(str(image), ps.cv2.IMREAD_GRAYSCALE)[:512] 
        
images = get_images(paths)  

#returns the dw motion object
data = ps.calculate_domain_motion(images, measurmentType, binarize)

# outputs parallel lines with outline as spread/delta. A line is a tuple of stating and ending point
parallels = ps.parallel_lines(measurmentType.point2, measurmentType.point1,
                              measurmentType.outline)

# returns a parallels
inter = [data.get_intersect((x[0], x[1])) for x in parallels]

# just to check the type of each itersection point
inter_type = [[str(type(x)) for x in inte] for inte in inter]

#making every intersection a point
new_inter = [[list(x.geoms)[0] if not (isinstance(x, ps.Pnt) or isinstance(x, ps.lstr)) 
             else x if isinstance(x, ps.Pnt) else ps.Pnt(x.coords[0]) if x.coords 
             else None for x in inte] for inte in inter]
# finding the intersection between
res = [[x.distance(ps.Pnt(y[0])) if x else x for x in new_inte] for y, new_inte in zip(parallels, new_inter)]
#finding the mean from intersection of all parallel lines
res = np.array(res, dtype = float).mean(0)
# video is 16 frames per second hence the 16 here
plt.plot(np.arange(len(res))/16, res*ps.x20)

result_dataframe = pd.DataFrame((np.arange(len(res))/16, res*ps.x20)).T
result_dataframe.to_clipboard(header = False, index = False)
#%%
# logic of the list comperhension
new_inter = []
for ii, x in enumerate(inter):
    if not (isinstance(x, ps.Pnt) or isinstance(x, ps.lstr)):
        new_inter.append(list(x.geoms)[0])
    elif isinstance(x, ps.Pnt):
        new_inter.append(x)
    else:
        if x.coords:
            print(ii)
            new_inter.append(ps.Pnt(x.coords[0]))
        else:
            new_inter.append(None)
            



            