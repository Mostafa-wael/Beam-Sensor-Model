# Beam-Sensor-Model
This is a quick and simple implementation of a beam sensor model. The beam sensor model is a simple model for simulating laser scanners. It is based on the [sensor model](http://wiki.ros.org/sensor_model) package, but uses a beam model instead of a ray model. The beam model is more suitable for simulating laser scanners, which are often used in robotics.
>> The results are saved in the `Output` directory.

## Part 1: Beam Readings for a certain pose
### Static

![Beam Sensor Model](Output/req1.png)

### Dynamic

https://user-images.githubusercontent.com/56788883/205514430-a6efdb5c-16e9-4c3a-b4c8-913ad375c849.mp4


## Part 2: Estimating the pose of the robot based on the beam readings

### Robots pose = (20, 20, 0)

![pose estimation](Output/output.png)

### More clear images after applying erosion and dilation

![pose estimation](Output/outputDilated.png)

![pose estimation](Output/outputEroted.png)

### Final likelihood with the map

![pose estimation](Output/outputTotalWithDilation.png)


### For the pose estimation, a multi-threaded code was used
![cpu usage](Output/CPU.png)

