# Imports
from utilities import showStaticMap, getColoredInitialMap, getBinaryInitialMap, getInitialRobotPose, showAnimatedMap

if __name__ == '__main__':    
    showStaticMap(getColoredInitialMap(), getInitialRobotPose())
    showAnimatedMap(getInitialRobotPose())
             