from glob import glob

import matplotlib.pyplot as plt  
import matplotlib as mpl  
import numpy as np  
import openslide  
from openslide import open_slide  
from openslide.deepzoom import DeepZoomGenerator  
import pandas as pd

from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_dilation  
from skimage.color import rgb2gray  
from skimage.morphology import closing, binary_closing, disk, remove_small_holes, dilation, remove_small_objects  
from skimage import color, morphology, filters, exposure, feature



plt.rcParams['figure.figsize'] = (10, 6)

files = glob("TCGA-2A-A8VL-01A-02-TS2.AFBBB2D5-39E6-434A-B6E5-779DD8217DCD.svs")
files
slide_num = 1
slide = open_slide(files[slide_num-1])

tile_size = 1024  
tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=False)  
# overlap adds pixels to each side
# See how many tiles there are for each level of magnification.
tiles.level_tiles  
#choose tiles you want to look at. You can change around 
#the coordinates to get the tile you are looking for.
#This is where OpenSlide helps.
tile = tiles.get_tile(tiles.level_count-18, (30, 15))  
tile  