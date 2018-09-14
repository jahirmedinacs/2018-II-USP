import pandas as pd
import numpy as np

import matplotlib as m_plt
from matplotlib import pyplot as plt

from copy import copy
from pprint import pprint

import sys
import os

def dummy_DataFrame_plotter(dataFrame_dummy,
                            plot_size=(10,10),
                            grid_subplot=[3,3],
                            mixed_style=False,
                            plot_style='', 
                            grid=False, 
                            have_lines=False,
                            line_style='r',
                            constant_values=[], 
                            verbose=False):
    
    plt.figure(figsize=plot_size)
    
    for column_id in range(dataFrame_dummy.shape[1]):
        
        temp_obj = plt.subplot(grid_subplot[0],grid_subplot[1], column_id + 1)
        
        temp_values = dataFrame_dummy.iloc[:, column_id]
        temp_obj.set_title(dataFrame_dummy.columns[column_id] + "    [Id   : {:d}]".format(column_id))
        
        if mixed_style:
            for temp_style in plot_style:
                temp_obj.plot(temp_values, temp_style)
        else:
            temp_obj.plot(temp_values, plot_style)
        
        if grid:
            temp_obj.grid()
        else:
            pass
        
        if have_lines:
            try:
                for temp_constant in constant_values[column_id]:
                    line_x = [dataFrame_dummy.index.values.min(), dataFrame_dummy.index.values.max()]
                    line_y = [temp_constant, temp_constant]
                    
                    temp_obj.plot(line_x, line_y, line_style)
            except:
                line_x = [dataFrame_dummy.index.values.min(), dataFrame_dummy.index.values.max()]
                line_y = [constant_values[column_id], constant_values[column_id]]

                temp_obj.plot(line_x, line_y, line_style)
            else:
                pass
            
        
        if verbose:
            temp_shape = dataFrame_dummy.iloc[:, column_id].shape 
            print("Column ID {:d} elements : [{:f}]".format(column_id, temp_shape[0]))
    
    return plt

def dummy_Histogram_plotter(histogram_container,
                            plot_size=(15,15),
                            grid_subplot=[3,3]):
    
    plt.figure(figsize=plot_size)
    
    for plot_id in range(len(histogram_container)):
        
        temp_obj = plt.subplot(grid_subplot[0],grid_subplot[1], plot_id + 1)
        
        temp_obj.set_title("[Id   : {:d}]".format(plot_id))
        
        temp_bins = histogram_container[plot_id][1]
        
        temp_color = m_plt.colors.Colormap(name="temp_color", N=len(temp_bins))
        
        temp_obj.bar( height=histogram_container[plot_id][0], 
                     x=temp_bins
                    )
    return plt