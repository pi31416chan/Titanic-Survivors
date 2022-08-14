# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Classes



# Functions
def plot_boxplot(data:pd.DataFrame,ticks_label:list,vert:bool=True,
                title:str=None,xlabel:str=None,ylabel:str=None,
                figsize:tuple=(10,6)):
    '''
    Quick plot of the boxplot of DataFrame or Series object.
    '''
    plt.figure(figsize=figsize)
    plt.boxplot(data,vert=vert)
    if vert: plt.xticks(ticks=np.arange(len(ticks_label))+1,labels=ticks_label)
    elif not vert: plt.yticks(ticks=np.arange(len(ticks_label))+1,labels=ticks_label)
    if title is not None: plt.title(title)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)       
    plt.show()