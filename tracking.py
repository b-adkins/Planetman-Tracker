import random

import cv2
import munkres
import numpy as np

class Contact:
    """
    A tracked object. Named with radar terminology
    
    Heavily borrowed from https://github.com/angusleigh/leg_tracker , specifically joint_leg_tracker.py
    """
    
    new_leg_id_num = 1

    def __init__(self, x, y, now, confidence, in_free_space, scan_frequency=7.5): 
        """
        Constructor
        """        
        self.id_num = Contact.new_leg_id_num
        Contact.new_leg_id_num += 1
        self.colour = (random.random(), random.random(), random.random())
        self.last_seen = now
        self.seen_in_current_scan = True
        self.times_seen = 1
        self.confidence = confidence
        self.dist_travelled = 0.
        self.deleted = False
        self.in_free_space = in_free_space

        # People are tracked via a constant-velocity Kalman filter with a Gaussian acceleration distrubtion
        # Kalman filter params were found by hand-tuning. 
        # A better method would be to use data-driven EM find the params. 
        # The important part is that the observations are "weighted" higher than the motion model 
        # because they're more trustworthy and the motion model kinda sucks
        delta_t = 1./scan_frequency
        if scan_frequency > 7.49 and scan_frequency < 7.51:
            std_process_noise = 0.06666
        elif scan_frequency > 9.99 and scan_frequency < 10.01:
            std_process_noise = 0.05
        elif scan_frequency > 14.99 and scan_frequency < 15.01:
            std_process_noise = 0.03333
        else:
            print "Scan frequency needs to be either 7.5, 10 or 15 or the standard deviation of the process noise needs to be tuned to your scanner frequency"
        std_pos = std_process_noise
        std_vel = std_process_noise
        std_obs = 0.1
        var_pos = std_pos**2
        var_vel = std_vel**2
        # The observation noise is assumed to be different when updating the Kalman filter than when doing data association
        var_obs_local = std_obs**2 
        self.var_obs = (std_obs + 0.4)**2

        self.filtered_state_means = self.state = np.array([x, y, 0, 0], np.float32).reshape((4, 1))
        self.pos_x = x
        self.pos_y = y
        self.vel_x = 0.0
        self.vel_y = 0.0

        self.filtered_state_covariances = 0.5*np.eye(4, dtype=np.float32) 

        self.kf = cv2.KalmanFilter(4, 4)
        
        # Commented out until I translate pixels to actual units of distance
        # # Constant velocity motion model
        # self.kf.transitionMatrix = np.array([[1, 0, delta_t,        0],
        #                                      [0, 1,       0,  delta_t],
        #                                      [0, 0,       1,        0],
        #                                      [0, 0,       0,        1]], np.float32)
        # 
        # # Oberservation model. Can observe pos_x and pos_y (unless person is occluded). 
        # self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
        #                                       [0, 1, 0, 0],
        #                                       [1, 0, 0, 0],
        #                                       [0, 1, 0, 0]], np.float32)
        #                                       
        # self.kf.processNoiseCov = np.array([[var_pos,       0,       0,       0],
        #                                     [      0, var_pos,       0,       0],
        #                                     [      0,       0, var_vel,       0],
        #                                     [      0,       0,       0, var_vel]], np.float32)
        # 
        # self.kf.measurementNoiseCov =  var_obs_local*np.eye(4, dtype=np.float32)
        
        delta_t = 1/60.
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0], [1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kf.processNoiseCov = np.array([[1,0,0.2,0],[0,1,0,0.2],[0.2,0,1,0],[0,0.2,0,1]],np.float32) * 0.03
        self.kf.measurementNoiseCov = np.array([[1, 0.5, 0, 0], [0.5, 1, 0, 0], [1, 0.5, 0, 0], [0.5, 1, 0, 0]], np.float32) * 0.4       
        
    def update(self, observations):
        """
        Update our tracked object with new observations
        """
        self.kf.correct(observations)
        self.filtered_state_means = self.kf.predict()

        # Keep track of the distance it's travelled 
        # We include an "if" structure to exclude small distance changes, 
        # which are likely to have been caused by changes in observation angle
        # or other similar factors, and not due to the object actually moving
        delta_dist_travelled = ((self.pos_x - self.filtered_state_means[0])**2 + (self.pos_y - self.filtered_state_means[1])**2)**(1./2.) 
        if delta_dist_travelled > 0.01: 
            self.dist_travelled += delta_dist_travelled
        
        self.state = self.filtered_state_means
        self.pos_x = self.filtered_state_means[0]
        self.pos_y = self.filtered_state_means[1]
        self.vel_x = self.filtered_state_means[2]
        self.vel_y = self.filtered_state_means[3]
