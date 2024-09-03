# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 23:59:35 2024

@author: Rakhul Raj
"""
import threading
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
from typing import List
import psutil
import numpy as np
import jsonpickle
# from werkzeug.serving import make_server # basic wsgi sever not for production
from waitress.server import create_server
from flask import Flask, request, jsonify
from typing import TYPE_CHECKING, Any

import processing as ps

# for typing
if TYPE_CHECKING:
    from custom_widgets import MainWindow
    from multiprocessing.pool import Pool as MPool

class MyFlaskApp:

    def __init__(self, window: 'MainWindow'):

        self.app = Flask(__name__)

        self.window = window
        self.app.add_url_rule('/api/avg_dis', 'get_disp_avg', self.get_disp_avg, methods=['POST'])
        self.app.add_url_rule('/api/dis', 'get_disp', self.get_dis, methods=['POST'])
        self.app.add_url_rule('/api/edge', 'get_edge', self.get_edge, methods=['POST'])
        self.app.add_url_rule('/api/edge_from_path', 'get_edge_from_path', self.get_edge_from_path, methods=['POST'])
        self.app.add_url_rule('/api/current_state', 'get_curr_state', self.get_curr_state)
        self.app.add_url_rule('/api/current_avg_dis', 'get_curr_dis_avg', self.get_curr_dis_avg)
        self.app.add_url_rule('/api/current_dis', 'get_curr_dis', self.get_curr_dis)
        self.app.add_url_rule('/api/current_edge', 'get_curr_edge', self.get_curr_edge)
        
        # self.app.add_url_rule('/api/current_dis', 
        
        # self.app.add_url_rule('/shutdown', 'shutdown', self.shutdown, methods=['POST'])
        self.app.add_url_rule('/', 'home', self.home)
        
        self.pool: 'MPool'= None
        
        self.server_started = False
        self.server = None
        self.thread = None

    def get_disp_avg(self):
        data = request.get_json()

        displacement = self.process_images(data,)
        displacement = [x for x in displacement if x]
        displacement = np.array([x for y in displacement for x in y])
        avg_disp: float = np.nanmean(displacement)

        return jsonify(avg_disp)
    
    def get_dis(self):

        data = request.get_json()
        displacement = self.process_images(data,)

        return displacement

    def get_edge(self):
        data = request.get_json()

        # Get the list of images from received data
        keys = (x for x in data.keys() if x.lower().startswith('image'))
        keys = sorted(keys, key=lambda x: int(x[5:]))
        images = [np.array(data[x], dtype = np.uint8) for x in keys]


        measType = ps.Meas_Type.from_window(self.window)
        binarize = ps.Binarize_Type.from_window(self.window)


        edge_image: np.ndarray[np.uint8] = self.pool.apply(ps.get_edge,
                                                           kwds= dict(images=images,
                                                                        measType=measType,
                                                                        binarize=binarize))

        return edge_image.tolist()
    
    def get_edge_from_path(self):
        data = request.get_json()
        measType = ps.Meas_Type.from_window(self.window)
        binarize = ps.Binarize_Type.from_window(self.window)

        edge_img = self.pool.apply(get_edge_from_path_worker, args = (data, measType, binarize))

        return edge_img.tolist()
    
    def get_curr_state(self,):

        measType = ps.Meas_Type.from_window(self.window)
        binarize = ps.Binarize_Type.from_window(self.window)
        return jsonify({
        'measType' : jsonpickle.encode(measType),
        'binarize' : jsonpickle.encode(binarize)
        })
    
    def get_curr_dis_avg(self):
        data = {f'image{ii}':x for ii, x in enumerate(self.window.images)}

        displacement = self.process_images(data,)
        displacement = [x for x in displacement if x]
        displacement = np.array([x for y in displacement for x in y])
        avg_disp: float = np.nanmean(displacement)

        return jsonify(avg_disp)
    
    def get_curr_dis(self):

        data = {f'image{ii}':x for ii, x in enumerate(self.window.images)}
        displacement = self.process_images(data,)
        
        return jsonify(displacement)
    
    def get_curr_edge(self):
        data = {f'image{ii}':x for ii, x in enumerate(self.window.images)}

        # Get the list of images from received data
        keys = (x for x in data.keys() if x.lower().startswith('image'))
        keys = sorted(keys, key=lambda x: int(x[5:]))
        images = [data[x] for x in keys]
        
        measType = ps.Meas_Type.from_window(self.window)
        binarize = ps.Binarize_Type.from_window(self.window)

        edge_image: np.ndarray[np.uint8] = self.pool.apply(ps.get_edge,
                                                           kwds= dict(images=images,
                                                                        measType=measType,
                                                                        binarize=binarize))

        return edge_image.tolist()


    def home(self):
        # return "MODVORTEx is running in this port"
        return [*request.environ.keys()]

    def process_images(self, data: dict):
        
        # Get the list of images from received data
        keys = (x for x in data.keys() if x.lower().startswith('image'))
        keys = sorted(keys, key=lambda x: int(x[5:]))
        images = [data[x] for x in keys]

        # Define the measurement type and binarization type objects
        measType = ps.Meas_Type.from_window(window=self.window)
        binarize = ps.Binarize_Type.from_window(window=self.window)

        # displacements: List[List[float]] = ps.calculate_motion_displace(images=images,
        #                                                         measType=measType,
        #                                                         binarize=binarize)

        displacements: List[List[float]] = self.pool.apply(ps.calculate_motion_displace,
                                                           kwds= dict(images=images,
                                                                        measType=measType,
                                                                        binarize=binarize))
        return displacements

    def run_server(self,):
        self.pool = Pool(processes= psutil.cpu_count(logical=False))
        self.server = create_server(self.app, host="localhost", port=5454, threads=8)
        self.thread = threading.Thread(target=self.server.run)
        self.thread.start()
        self.server_started = True

    def close_server(self):
        if self.server_started:
            self.server.close()
            self.thread.join()
            self.server_started = False
            self.pool.close()

def get_edge_from_path_worker(data : str, measType:ps.Meas_Type, binarize: ps.Binarize_Type):
        
        images = [ps.load_image(str(p), measType) for p in Path(data).glob('*.png')]
        edge_img = ps.get_edge(images, binarize, measType)

        return edge_img  
    
    # def run_server(self,):
    #     self.server = make_server('localhost', 5000, self.app)
    #     self.thread = threading.Thread(target=self.server.serve_forever)
    #     self.thread.start()
    #     self.server_started = True

    # def close_server(self):
    #     if self.server_started:
    #         self.server.shutdown()
    #         self.thread.join()
    #         self.server.server_close()
    #         self.server_started = False
    

    # def shutdown(self):
    #     data = request.get_json()
    #     if data.get('secret_key') == self.secret_key:
    #         func = request.environ.get('werkzeug.server.shutdown')
    #         if func is None:
    #             raise RuntimeError('Not running with the Werkzeug Server')
    #         func()
    #         return 'Server shutting down...'
    #     else:
    #         return 'Unauthorized', 401
