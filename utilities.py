# Imports
import numpy as np
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
#######################################################
# Constants
LASER_ANGLE_RANGE = 125 # to the left and the right(in degrees)
NUMBER_OF_LASER_RAYS = 2 * LASER_ANGLE_RANGE // 2 # 2 degrees per ray
LASER_MAX_RANGE = 1200//4 # (in pixels)
PIXEL_WIDTH = 4
PIXEL_HIGHT = 4
MAP_SIZE = (400, 680) # where the first dimension if for the y-axis and the second dimension (in pixels)
#######################################################
# Read Data
def getColoredInitialMap():
    # Read Map.jpg using skimage
    map = io.imread('Map.jpg')
    # Convert to grayscale
    map = np.mean(map, axis=2)
    # Apply threshold
    map = map < 128
    # Convert to 8-bit integer
    map = map * 255
    return map
def getBinaryInitialMap():
    # Read Map.jpg using skimage
    map = io.imread('Map.jpg')
    # Convert to grayscale using sklearn
    map = rgb2gray(map)
    # convert zeros to 0.1 to avoid division by zero
    map[map == 0] = 0.1
    return 1 - map
def getInitialRobotPose():
    robotPose = np.array([10, 180, 0]) # x-coordinates, y-coordinates, angle with the x-axis in degree
    return robotPose
#######################################################
# Drawing Functions
def drawRobot(map, robotPose):
    mapY, mapX, _ = robotPose
    if (mapX >= 0 and mapX < map.shape[0] and mapY >= 0 and mapY < map.shape[1]):
        map[mapX, mapY] = 200
    return map
def drawLaserLines(map, robotPose):
    robotX, robotY, robotTheta = robotPose 
    for theta in range(robotTheta - LASER_ANGLE_RANGE, robotTheta + LASER_ANGLE_RANGE, 2):
        # laser range, laser step
        r, dr = LASER_MAX_RANGE, 1
        for d in range(0, r, dr):
            # update the laser end point
            mapY = int(robotX + d * np.cos(theta * np.pi / 180)) # angles are in degrees
            mapX = int(robotY + d * np.sin(theta * np.pi / 180)) # angles are in degrees
            # if the laser is inside the map
            if (mapX >= 0 and mapX < map.shape[0] and mapY >= 0 and mapY < map.shape[1]):
                # Check if the laser hits an obstacle
                if (map[mapX, mapY] == 255): # revered as the origin is in the lower left corner
                    d = r # no need to check further points
                    break
                else:
                    map[mapX, mapY] = 100  # color the laser 
    return map
#######################################################
def getFigure():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.clear()  # Clears the figure to update the line, point, title, and axes
    # Adding Figure Labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return fig, ax

def showStaticMap(map, robotPose):
    map = drawRobot(drawLaserLines(map, robotPose), robotPose)
    fig, ax = getFigure()
    # Plot the robot
    plt.scatter(robotPose[0], robotPose[1], color='r')
    ax.set_title("Robot Pose: (%2d, %2d, %2d)" % (robotPose[0], robotPose[1], robotPose[2]))
    # plot the robot orientation as a line
    plt.plot(
            [robotPose[0], robotPose[0] + 10 * np.cos(robotPose[2] * np.pi / 180)], 
            [robotPose[1], robotPose[1] + 10 * np.sin(robotPose[2] * np.pi / 180)], 
            color='black')
    plt.imshow(map)
    plt.show()
def showAnimatedMap(robotPose):
    def animate(i):
        newRobotPose = robotPose.copy()
        newRobotPose[0] += i
        map = drawRobot(drawLaserLines(getColoredInitialMap(), newRobotPose), newRobotPose)
        ax.clear() 
        plt.scatter(newRobotPose[0], newRobotPose[1], color='r')
        # plot the robot orientation as a line
        plt.plot(
                [newRobotPose[0], newRobotPose[0] + 10 * np.cos(newRobotPose[2] * np.pi / 180)], 
                [newRobotPose[1], newRobotPose[1] + 10 * np.sin(newRobotPose[2] * np.pi / 180)], 
                color='black')
        ax.set_title("Robot Pose: (%2d, %2d, %2d)" % (newRobotPose[0], newRobotPose[1], newRobotPose[2]))
        plt.imshow(map)
    # initialize the figure
    fig, ax = getFigure()
    # animation
    anim = animation.FuncAnimation(fig, animate, init_func=getColoredInitialMap,
                               frames=MAP_SIZE[1], interval=1, blit=True,)
    plt.show()
    print("Saving the animation")
    # anim.save('map.gif', dpi=80, writer='pillow', fps=20)
    print("Done")
#######################################################
# Get rays from pose
def getRays(map, robotPose):
    laserRays = [LASER_MAX_RANGE]* NUMBER_OF_LASER_RAYS  # the default is the max range of rays
    robotX, robotY, robotTheta = robotPose 
    for rayIdx, theta in enumerate(range(robotTheta - LASER_ANGLE_RANGE, robotTheta + LASER_ANGLE_RANGE, 2)):
        # laser range, laser step
        r, dr = LASER_MAX_RANGE, 1
        for d in range(0, r, dr):
            # update the laser end point
            mapY = int(robotX + d * np.cos(theta * np.pi / 180))
            mapX = int(robotY + d * np.sin(theta * np.pi / 180))
            # if the laser is inside the map
            if (mapX >= 0 and mapX < map.shape[0] and mapY >= 0 and mapY < map.shape[1]):
                # Check if the laser hits an obstacle
                if (map[mapX, mapY] == 255): # revered as the origin is in the lower left corner
                    laserRays[rayIdx] = d
                    d = r # no need to check further points
                    break
                else:
                    map[mapX, mapY] = 100  # color the laser 
    return laserRays
#######################################################
# Likelihood Field
def getEndPointProbability(likelihood, rays, pointX, pointY, theta):
    probability = 1 # initialize the probability
    for rayIdx, angle in enumerate(range(theta-125, theta+125, 2)):  
        # update the laser end point
        mapY = int(pointX + rays[rayIdx] * np.cos(angle * np.pi / 180))
        mapX = int(pointY + rays[rayIdx] * np.sin(angle * np.pi / 180))
        # if the laser is inside the map
        if (mapX >= 0 and mapX < likelihood.shape[0] and mapY >= 0 and mapY < likelihood.shape[1]):
                probability*= likelihood[mapX, mapY]
        else: # if the laser is outside the map
            probability*=0.1      
    return probability


def getLikelihood(likelihood, rays):
    map =  np.zeros(likelihood.shape)
    for x in range(0, likelihood.shape[0]//10, 2):
        for y in range(0, likelihood.shape[1]//10,2):
            for theta in range(0, 360, 35):
                map[x, y] = max(getEndPointProbability(likelihood, rays, x, y, theta), map[x, y])
    return map
# make the getLikelihood function runs in parallel
import multiprocessing as mp
def getLikelihoodParallel(likelihood, rays):
    pool = mp.Pool(mp.cpu_count())
    map =  np.zeros(likelihood.shape)
    for x in tqdm(range(0, likelihood.shape[0]//10, 2), desc="Calculating Likelihood"):
        for y in range(0, likelihood.shape[1]//10,2):
            for theta in range(0, 360, 35):
                map[x, y] = max(pool.apply(getEndPointProbability, args=(likelihood, rays, x, y, theta)), map[x, y])
    pool.close()
    return map

#######################################################
#######################################################