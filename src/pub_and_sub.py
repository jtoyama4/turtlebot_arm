#!/usr/bin/env python

import math
import rospy
from std_msgs.msg import Float64
from dynamixel_msgs.msg import JointState
import sys
import numpy as np
class RL(object):
    def __init__(self):
        self.value=0.0

        rospy.init_node('pub_and_sub')

        self.pub = rospy.Publisher('/pan4_controller/command', Float64, queue_size = 10)
        rospy.Subscriber('/pan4_controller/state', JointState, self.update_state)
        self.angle = 0
        self.fangle = 0
        self.state_dim = 1
        self.theta = np.random.rand(self.state_dim)
        self.sigma = np.array(3.0)
        self.w = np.random.rand(self.state_dim)
        self.alpha = 0.5
        self.gamma = 0.95
        self.velocity = 0.0
        
    def update_state(self, msg):
        self.angle = msg.current_pos
        self.fangle = msg.current_pos - 2.5
        #self.velocity = msg.velocity

    def get_action(self, state):
        mu = np.sum(self.theta * state) / self.state_dim
        print 1+np.exp(-self.sigma)
        x = np.random.normal(mu, 1.0/(1+np.exp(-self.sigma)))
        print 1+np.exp(-self.sigma)
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0
        if state > 0.0:
            return x
        else:
            return -x

    def Value(self,state):
        b = np.array([2.5]).astype("float64")
        v = sum(self.w * (state - b))
        return v

    def update_model(self, r, state, a):
        TDErr = max(min((r + self.gamma * self.Value(state) - self.Value(self.prev_state)), 0.9), -0.9)

        mu = np.sum(self.theta * state)/self.state_dim
        print "TDErr ", TDErr, "state ", state, "action ", a, "reward ", r
        state = np.array(state)
        self.theta += self.alpha * TDErr * (a - mu) * state / self.state_dim
        tmp = self.sigma + self.alpha * TDErr * ((a-mu)**2 - self.sigma**2)
        tmp = min(max(tmp, -10.0), 10.0)
        self.sigma = tmp
        self.w += self.alpha * TDErr * state
        
    def get_reward(self):
        reward = -math.cos(math.pi*(self.angle * 0.333 + 0.1666))
        return reward
        
    def torque_controller(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            self.prev_state=[self.fangle, self.velocity]
            action = self.get_action(self.prev_state)
            self.pub.publish(action)
            rate.sleep()
            reward = self.get_reward()
            #state = [self.angle, self.velocity]
            state = [self.fangle]
            self.update_model(reward, state, action)

if __name__ == '__main__':
    try:
        rl = RL()
        rl.torque_controller()
    except rospy.ROSInterruptException:
        pass
