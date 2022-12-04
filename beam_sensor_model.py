# Imports
from utilities import showStaticMap, getColoredInitialMap, getBinaryInitialMap, getInitialRobotPose, showAnimatedMap
from utilities import  getRays, getLikelihood
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import matplotlib.animation as animation
from tqdm import tqdm

if __name__ == '__main__':    
    showStaticMap(getColoredInitialMap(), getInitialRobotPose())
    showAnimatedMap(getInitialRobotPose())
             