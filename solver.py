import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage import io
import matplotlib.animation as animation


# Constants
LASER_ANGLE_RANGE = 125 # to the left and the right(in degrees)
LASER_MAX_RANGE = 1200//4 # (in pixels)
PIXEL_WIDTH = 4
PIXEL_HIGHT = 4
MAP_SIZE = (400, 680) # where the first dimension if for the y-axis and the second dimension (in pixels)
def getMap():
    # Read Map.jpg using skimage
    map = io.imread('Map.jpg')
    # Convert to grayscale
    map = np.mean(map, axis=2)
    # Apply threshold
    map = map < 128
    # Convert to 8-bit integer
    map = map * 255
    return map
def getInitialRobotPose():
    robotPose = np.array([10, 180, 0]) # x-coordinates, y-coordinates, angle with the x-axis in degree
    return robotPose
def drawRobot(map, robotPose):
    mapY, mapX, _ = robotPose
    if (mapX >= 0 and mapX < map.shape[0] and mapY >= 0 and mapY < map.shape[1]):
        map[mapX, mapY] = 200
    return map
def drawLaserLines(map, robotPose):
    robotX, robotY, robotTheta = robotPose 
    for theta in range(robotTheta - LASER_ANGLE_RANGE, robotTheta + LASER_ANGLE_RANGE, 2):
        # laser range
        r = LASER_MAX_RANGE
        # laser step
        dr = 1
        for d in range(0, r, dr):
            # update the laser end point
            mapY = int(robotX + d * np.cos(theta * np.pi / 180) )
            mapX = int(robotY + d * np.sin(theta * np.pi / 180) )
            # if the laser is inside the map
            if (mapX >= 0 and mapX < map.shape[0] and mapY >= 0 and mapY < map.shape[1]):
                # Check if the laser hits an obstacle
                if (map[mapX, mapY] == 255): # revered as the origin is in the lower left corner
                    d = r # no need to check further points
                    break
                else:
                    map[mapX, mapY] = 100  # color the laser 
    return map
def getFigure():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.clear()  # Clears the figure to update the line, point, title, and axes
    # Adding Figure Labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return fig, ax
def showStaticMap():
    robotPose = getInitialRobotPose()
    map = drawRobot(drawLaserLines(getMap(), robotPose), robotPose)
    fig, ax = getFigure()
    # Plot the robot
    plt.scatter(robotPose[0], robotPose[1], color='r')
    ax.set_title("Robot Pose: (%2d, %2d, %2d)" % (robotPose[0], robotPose[1], robotPose[2]))
    plt.imshow(map)
    plt.show()
def showAnimatedMap():
    def animate(i):
        robotPose = getInitialRobotPose()
        robotPose[0] += i
        map = drawRobot(drawLaserLines(getMap(), robotPose), robotPose)
        # map = drawRobot(map, robotPose)
        ax.clear() 
        plt.scatter(robotPose[0], robotPose[1], color='r')
        ax.set_title("Robot Pose: (%2d, %2d, %2d)" % (robotPose[0], robotPose[1], robotPose[2]))
        plt.imshow(map)
    # initialize the figure
    fig, ax = getFigure()
    # animation
    anim = animation.FuncAnimation(fig, animate, init_func=getMap,
                               frames=MAP_SIZE[1], interval=1, blit=True,)
    plt.show()
    # anim.save('map.gif', dpi=80, writer='pillow')

if __name__ == '__main__':    
    showStaticMap()
    showAnimatedMap()                