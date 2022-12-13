#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FastSLAM algorithm for range-bearing landmark observations.
"""

from __future__ import print_function, division

import copy
import random

from math import sqrt, pi, exp, sin, cos, atan, atan2

from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np

# Chi square inverse cumulative distribution function for alpha = 0.95
# and 2 degrees of freedom (see lookup table in statistics textbook)
CHI_SQUARE_INV_95_2 = 5.99146


class Odometry(object):
    """Represents an odometry command.

    - r1: initial rotation in radians counterclockwise
    - t: translation in meters
    - r2: final rotation in radians counterclockwise
    """
    def __init__(self, r1, t, r2):
        self.r1 = r1
        self.t = t
        self.r2 = r2

    def __str__(self):
        return "Odometry(r1 = {0} rad, t = {1} m, r2 = {2} rad)".format(self.r1, self.t, self.r2)


class SensorMeasurement(object):
    """Represents one or more range and bearing sensor measurements.

    - id: ID of the landmark
    - z_range: measured distance to the landmark in meters
    - z_bearing: measured angle towards the landmark in radians
    """

    def __init__(self, landmark_id, z_range, z_bearing):
        self.landmark_id = landmark_id
        self.z_range = z_range
        self.z_bearing = z_bearing

    def __str__(self):
        return "SensorMeasurement(landmark_id = {0}, z_range = {1} m, z_bearing = {2} rad)".format(
            self.landmark_id, self.z_range, self.z_bearing)


class TimeStep(object):
    """Represents a data point consisting of an odometry command and a list of sensor measurements.

    - odom: Odometry measurement
    - sensor: List of landmark observations of type SensorMeasurement
    """

    def __init__(self, odom, sensor):
        self.odom = odom
        self.sensor = sensor

    def __str__(self):
        return "TimeStep(odom = {0}, sensor = {1})".format(self.odom, self.sensor)


def read_world(filename):
    """Reads the world definition and returns a structure of landmarks.

    filename: path of the file to load
    landmarks: structure containing the parsed information

    Each landmark contains the following information:
    - id : id of the landmark
    - x  : x-coordinate
    - y  : y-coordinate

    Examples:
    - Obtain x-coordinate of the 5-th landmark
      landmarks(5).x
    """
    landmarks = {}
    with open(filename) as world_file:
        for line in world_file:
            line_s = line.rstrip()  # remove newline character

            line_spl = line_s.split(' ')

            landmarks[int(line_spl[0])] = [float(line_spl[1]), float(line_spl[2])]

    return landmarks


def read_sensor_data(filename):
    """Reads the odometry and sensor readings from a file.

    filename: path to the file to parse

    The data is returned as an array of time steps, each time step containing
    odometry data and sensor measurements.
    
    Usage:
    - access the readings for timestep t:
      data[t]
      this returns a TimeStep object containing the odometry reading and all
      landmark observations, which can be accessed as follows
    - odometry reading at timestep t:
      data[i].odometry
    - sensor reading at timestep i:
      data[i].sensor

    Odometry readings have the following fields:
    - r1 : initial rotation in radians counterclockwise
    - t  : translation in meters
    - r2 : final rotation in radians counterclockwise
    which correspond to the identically labeled variables in the motion
    mode.

    Sensor readings can again be indexed and each of the entries has the
    following fields:
    - landmark_id : id of the observed landmark
    - z_range     : measured range to the landmark in meters
    - z_bearing   : measured angle to the landmark in radians

    Examples:
    - Translational component of the odometry reading at timestep 10
      data[10].odometry.t
    - Measured range to the second landmark observed at timestep 4
      data[4].sensor[1].z_range
    """
    data = []
    sensor_measurements = []
    first_time = True
    odom = None
    with open(filename) as data_file:
        for line in data_file:
            line_s = line.rstrip()  # remove the new line character
            line_spl = line_s.split(' ')  # split the line
            if line_spl[0] == 'ODOMETRY':
                if not first_time:
                    data.append(TimeStep(
                        odom=odom,
                        sensor=sensor_measurements
                    ))
                    sensor_measurements = []
                first_time = False
                odom = Odometry(r1=float(line_spl[1]), t=float(line_spl[2]), r2=float(line_spl[3]))
            if line_spl[0] == 'SENSOR':
                sensor_measurements.append(SensorMeasurement(
                    landmark_id=int(line_spl[1]),
                    z_range=float(line_spl[2]),
                    z_bearing=float(line_spl[3])
                ))

    data.append(TimeStep(odom=odom, sensor=sensor_measurements))
    return data


def normalize_angle(angle):
    """Normalize the angle between -pi and pi"""

    while angle > pi:
        angle = angle - 2. * pi

    while angle < -pi:
        angle = angle + 2. * pi

    return angle


class Particle(object):
    """Particle for tracking a robot with a particle filter.

    The particle consists of:
    - a robot pose
    - a weight
    - a map consisting of landmarks
    """

    class LandmarkEKF(object):
        """EKF representing a landmark"""
        def __init__(self):
            self.observed = False
            self.mu = np.vstack([0, 0])  # landmark position as vector of length 2
            self.sigma = np.zeros((2, 2))  # covariance as 2x2 matrix

        def __str__(self):
            return "LandmarkEKF(observed  = {0}, mu = {1}, sigma = {2})".format(self.observed, self.mu, self.sigma)


    def __init__(self, num_particles, num_landmarks, noise):
        """Creates the particle and initializes location/orientation"""
        self.noise = noise

        # initialize robot pose at origin
        self.pose = np.vstack([0., 0., 0.])

        # initialize weights uniformly
        self.weight = 1.0 / float(num_particles)

        # Trajectory of the particle
        self.trajectory = []

        # initialize the landmarks aka the map
        self.landmarks = [self.LandmarkEKF() for _ in range(num_landmarks)]


    def prediction_step(self, odom):
        """Predict the new pose of the robot"""

        # append the old position
        self.trajectory.append(self.pose)

        # noise sigma for delta_rot1
        delta_rot1_noisy = random.gauss(odom.r1, self.noise[0])

        # noise sigma for translation
        translation_noisy = random.gauss(odom.t, self.noise[1])

        # noise sigma for delta_rot2
        delta_rot2_noisy = random.gauss(odom.r2, self.noise[2])

        # Estimate of the new position of the Particle
        x_new = self.pose[0] + translation_noisy * cos(self.pose[2] + delta_rot1_noisy)
        y_new = self.pose[1] + translation_noisy * sin(self.pose[2] + delta_rot1_noisy)
        theta_new = normalize_angle(self.pose[2] + delta_rot1_noisy + delta_rot2_noisy)

        self.pose = np.vstack([x_new, y_new, theta_new])


    def correction_step(self, sensor_measurements):
        """Weight the particles according to the current map of the particle and the landmark observations z.

        - sensor_measurements                : list of sensor measurements for the current timestep
        - sensor_measurements[i].landmark_id : observed landmark ID
        - sensor_measurements[i].z_range     : measured distance to the landmark in meters
        - sensor_measurements[i].z_bearing   : measured angular direction of the landmark in radians
        """

        # Construct the sensor noise matrix Q (2 x 2)
        Q_t = 0.1 * np.identity(2)

        robot_pose = self.pose

        # process each sensor measurement
        for measurement in sensor_measurements:
            # Get the EKF representing the landmark of the current observation
            landmark = self.landmarks[measurement.landmark_id]

            # The (2x2) EKF of the landmark is given by
            # its mean landmarks[landmark_id].mu
            # and by its covariance landmarks[landmark_id].sigma

            # If the landmark is observed for the first time:
            if not landmark.observed:
                # TODO: Initialize its position based on the measurement and the current Particle pose:

                # get the Jacobian
                [h, H] = self.measurement_model(landmark)

                # TODO: initialize the EKF for this landmark

                # Indicate that this landmark has been observed
                landmark.observed = True

            else:
                # get the expected measurement and the Jacobian
                [expected_z, H] = self.measurement_model(landmark)

                # TODO: compute the measurement covariance

                # TODO: calculate the Kalman gain

                # TODO: compute the error between the z and expected_z (remember to normalize the angle)

                # TODO: update the mean and covariance of the EKF for this landmark

                # TODO: compute the likelihood of this observation, multiply with the former weight
                # to account for observing several features in one time step


    def measurement_model(self, landmark_ekf):
        """Compute the expected measurement for a landmark and the Jacobian

        - landmark_ekf: EKF representing the landmark

        Returns a tuple (h, H) where
        - h = [expected_range, expected_range] is the expected measurement
        - H is the Jacobian.
        """
        # position (x, y) of the observed landmark_ekf
        landmark_x = landmark_ekf.mu[0]
        landmark_y = landmark_ekf.mu[1]

        # use the current state of the particle to predict the measurement
        expected_range = np.sqrt((landmark_x - self.pose[0]) ** 2 + (landmark_y - self.pose[1]) ** 2)
        angle = atan2(landmark_y - self.pose[1], landmark_x - self.pose[0]) - self.pose[2]
        expected_bearing = normalize_angle(angle)

        h = [expected_range, expected_bearing]

        # Compute the Jacobian H of the measurement z with respect to the landmark position
        H = np.zeros((2, 2))
        H[0, 0] = ((landmark_x - self.pose[0]) / expected_range)         # d_range / d_lx
        H[0, 1] = ((landmark_y - self.pose[1]) / expected_range)         # d_range / d_ly

        H[1, 0] = ((self.pose[1] - landmark_y) / (expected_range ** 2))  # d_bearing / d_lx
        H[1, 1] = ((landmark_x - self.pose[0]) / (expected_range ** 2))  # d_bearing / d_ly

        return (h, H)


def resample(particles):
    """Resample the set of particles.

    A particle has a probability proportional to its weight to get
    selected. A good option for such a resampling method is the so-called low
    variance sampling, Probabilistic Robotics page 109"""
    num_particles = len(particles)
    new_particles = []
    weights = [particle.weight for particle in particles]

    # normalize the weight
    sum_weights = sum(weights)
    weights = [weight / sum_weights for weight in weights]

    # the cumulative sum
    cumulative_weights = np.cumsum(weights)
    normalized_weights_sum = cumulative_weights[len(cumulative_weights) - 1]

    # check: the normalized weights sum should be 1 now (up to float representation errors)
    assert abs(normalized_weights_sum - 1.0) < 1e-5

    # initialize the step and the current position on the roulette wheel
    step = normalized_weights_sum / num_particles
    position = random.uniform(0, normalized_weights_sum)
    idx = 1

    # walk along the wheel to select the particles
    for i in range(1, num_particles + 1):
        position += step
        if position > normalized_weights_sum:
            position -= normalized_weights_sum
            idx = 1
        while position > cumulative_weights[idx - 1]:
            idx = idx + 1

        new_particles.append(copy.deepcopy(particles[idx - 1]))
        new_particles[i - 1].weight = 1 / num_particles

    return new_particles


class Plotter(object):
    '''Helper class for plotting the current state of the robot and map'''

    def __init__(self, data, particles, landmarks):
        self.data = data
        self.particles = particles
        self.landmarks = landmarks
        self.fig, self.ax = plt.subplots()
        plt.xlim(-2, 12)
        plt.ylim(-2, 12)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)

        # plot the ground truth landmark positions as black crosses
        landmarks_x = [coordinates[0] for _, coordinates in self.landmarks.items()]
        landmarks_y = [coordinates[1] for _, coordinates in self.landmarks.items()]
        self.plot_landmarks = self.ax.plot(landmarks_x, landmarks_y, 'k+', markersize=10, linewidth=5, animated=True)[0]

        # Plot the particles as green dots
        particles_x = [particle.pose[0] for particle in self.particles]
        particles_y = [particle.pose[1] for particle in self.particles]
        self.plot_particles = self.ax.plot(particles_x, particles_y, 'g.', animated=True)[0]

        # draw the best particle as a red circle
        self.plot_best_particle = Ellipse((0.0, 0.0), 0, 0, 0, color='r', fill=False, animated=True)
        self.ax.add_patch(self.plot_best_particle)

        # draw the trajectory as estimated by the currently best particle as a red line
        self.plot_trajectory = self.ax.plot([0.0], [0.0], 'r-', linewidth=3, animated=True)[0]


    def fast_slam(self, t):
        '''Executes one iteration of the prediction-correction-resampling loop of FastSLAM.

        - t: Frame number of the current frame, starting with 0.

        Returns the plot objects to be drawn for the current frame.
        '''
        # Perform filter update for each odometry-observation pair read from the data file.
        print('step {0}'.format(t))

        # Perform the prediction step of the particle filter
        for particle in self.particles:
            particle.prediction_step(self.data[t].odom)

        # Perform the correction step of the particle filter
        for particle in self.particles:
            particle.correction_step(self.data[t].sensor)

        # Generate visualization plots of the current state of the filter
        r = self.plot_state(self.data[t].sensor)

        # Resample the particle set
        # Use the "number of effective particles" approach to resample only when
        # necessary. This approach reduces the risk of particle depletion.
        # For details, see Section IV.B of
        # http://www2.informatik.uni-freiburg.de/~burgard/postscripts/grisetti05icra.pdf
        s = sum([particle.weight for particle in self.particles])
        neff = 1. / sum([(particle.weight/s) ** 2 for particle in self.particles])
        if neff < len(self.particles) / 2.:
            print ("resample")
            self.particles = resample(self.particles)

        return r


    def plot_state_init(self):
        '''Initializes the figure with the elements to be drawn'''
        return [self.plot_landmarks]


    def plot_state(self, sensor_measurements):
        """Visualizes the state of the FastSLAM algorithm.

        The resulting plot displays the following information:
        - map ground truth (black +'s)
        - currently best particle (red)
        - particle set in green
        - current landmark pose estimates (blue)
        - visualization of the observations made at this time step (line between Particle and landmark)
        """
        # update particle poses
        particles_x = [particle.pose[0] for particle in self.particles]
        particles_y = [particle.pose[1] for particle in self.particles]
        self.plot_particles.set_data(particles_x, particles_y)

        # determine the currently best particle
        weights = [particle.weight for particle in self.particles]
        index_of_best_particle = weights.index(max(weights))

        weight_sum = sum(weights)
        mean_pose = sum(particle.weight * particle.pose for particle in self.particles) / weight_sum
        cov_pose = sum(particle.weight * np.outer(particle.pose - mean_pose, particle.pose - mean_pose) for particle in self.particles) / weight_sum

        self.plot_best_particle.center = (mean_pose[0], mean_pose[1])
        (self.plot_best_particle.width, self.plot_best_particle.height, angle_rad) = self.get_ellipse_params(cov_pose)
        self.plot_best_particle.angle = angle_rad * 180. / pi

        # get trajectory points
        trajectory_x_list = [sum(particle.weight * particle.trajectory[i][0] for particle in self.particles) / weight_sum for i in range(0, len(self.particles[0].trajectory))]
        trajectory_y_list = [sum(particle.weight * particle.trajectory[i][1] for particle in self.particles) / weight_sum for i in range(0, len(self.particles[0].trajectory))]
        self.plot_trajectory.set_data(trajectory_x_list, trajectory_y_list)

        plots = [self.plot_landmarks, self.plot_particles, self.plot_best_particle, self.plot_trajectory]

        # draw the estimated landmark locations along with the ellipsoids
        for landmark in self.particles[index_of_best_particle].landmarks:
            if landmark.observed:
                bpx = landmark.mu[0]
                bpy = landmark.mu[1]
                plots.append(self.ax.plot(bpx, bpy, 'bo', markersize=3, animated=True)[0])
                [a, b, angle] = self.get_ellipse_params(landmark.sigma)
                angle_degrees = angle * 180. / pi
                e = Ellipse([bpx, bpy], a, b, angle_degrees, fill=False, animated=True)
                self.ax.add_patch(e)
                plots.append(e)

        # draw the observations as lines between the best particle and the observed landmarks
        for measurement in sensor_measurements:
            landmark_x = self.particles[index_of_best_particle].landmarks[measurement.landmark_id].mu[0]
            landmark_y = self.particles[index_of_best_particle].landmarks[measurement.landmark_id].mu[1]
            plots.append(plt.plot((landmark_x, mean_pose[0]), (landmark_y, mean_pose[1]), 'k', linewidth=1, animated=True)[0])

        return plots


    def get_ellipse_params(self, C):
        """Calculates unscaled half axes of the 95% covariance ellipse.

        C: covariance matrix
        alpha: confidence value

        Code from the CAS Robot Navigation Toolbox:
        http://svn.openslam.org/data/svn/cas-rnt/trunk/lib/drawprobellipse.m
        Copyright (C) 2004 CAS-KTH, ASL-EPFL, Kai Arras
        Licensed under the GNU General Public License, version 2
        """
        sxx = float(C[0, 0])
        syy = float(C[1, 1])
        sxy = float(C[0, 1])
        a = sqrt(0.5 * (sxx + syy + sqrt((sxx - syy) ** 2 + 4. * sxy ** 2)))  # always greater
        b = sqrt(0.5 * (sxx + syy - sqrt((sxx - syy) ** 2 + 4. * sxy ** 2)))  # always smaller

        # Remove imaginary parts in case of neg. definite C
        if not np.isreal(a):
            a = np.real(a)
        if not np.isreal(b):
            b = np.real(b)

        # Scaling in order to reflect specified probability
        a = a * sqrt(CHI_SQUARE_INV_95_2)
        b = b * sqrt(CHI_SQUARE_INV_95_2)

        # Look where the greater half axis belongs to
        if sxx < syy:
            swap = a
            a = b
            b = swap

        # Calculate inclination (numerically stable)
        if sxx != syy:
            angle = 0.5 * atan(2.*sxy / (sxx - syy))
        elif sxy == 0.:
            angle = 0.  # angle doesn't matter
        elif sxy > 0.:
            angle = pi / 4.
        elif sxy < 0.:
            angle = -pi / 4.
        return [a, b, angle]


def main():
    '''Main function of the program.

    This script calls all the required functions in the correct order.
    You can change the number of steps the filter runs for to ease the
    debugging. You should however not change the order or calls of any
    of the other lines, as it might break the framework.

    If you are unsure about the input and return values of functions you
    should read their documentation which tells you the expected dimensions.
    '''

    # Read world data, i.e. landmarks. The true landmark positions are not given to the Particle
    landmarks_truth = read_world('data/world.dat')

    # Read sensor readings, i.e. odometry and range-bearing sensor
    data = read_sensor_data('data/sensor_data.dat')

    # how many particles
    num_particles = 100

    # Get the number of landmarks in the map
    num_landmarks = len(landmarks_truth)

    noise = [0.005, 0.01, 0.005]

    # initialize the particles array
    particles = [Particle(num_particles, num_landmarks, noise) for _ in range(num_particles)]

    # set the axis dimensions
    plotter = Plotter(data, particles, landmarks_truth)

    anim = animation.FuncAnimation(plt.gcf(), plotter.fast_slam, frames=np.arange(0, len(data)), init_func=plotter.plot_state_init, interval=20, blit=True, repeat=False)
    if anim:
        plt.show()


if __name__ == '__main__':
    main()

