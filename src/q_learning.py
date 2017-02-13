#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
#from dynamixel_msgs.msg import JointState
import sys
import random
import numpy
import math
import time

class RL(object):
    def __init__(self):
        self.value=0.0
        self.states = [i * 0.1 for i in range(-15, 15)]
        #self.idx = random.sample([i for i in range(0, 30)], 1)[0]
        self.idx = 25
        rospy.init_node('q_learning')

        self.pub = rospy.Publisher('/tilt3_controller/command', Float64, queue_size=10)
        #rospy.Subscriber('/tilt3_controller/state', JointState, self.update_state)
        print self.states[self.idx]
        #self.pub.publish(self.states[self.idx])
        self.Q={}
        self.init_q()
        self.alpha = 0.1
        self.dis = 0.999
        self.e = 0.5

    def init_q(self):
        for i in range(30):
            self.Q[i] = [0,0]
            
    def action(self):
        self.next_idx = self.idx
        if self.move == 1:
            self.next_idx = self.idx + 1
            if self.next_idx == 30:
                self.next_idx = 29
        elif self.move == 0: 
            self.next_idx = self.idx - 1
            if self.next_idx == -1:
                self.next_idx = 0
        else:
            print "error"
                
    def update_state(self):
        angle = self.states[self.idx]
        
        choices = self.Q[self.idx]
        
        self.move = numpy.argmax(choices)
        if choices[0] == choices[1]:
            self.move = random.sample([0,1],1)[0]

        ran = random.random()
        if ran < self.e:
            if self.move == 0:
                self.move = 1
            else:
                self.move = 0
        
        self.action()

        new_angle = self.states[self.next_idx]
        self.pub.publish(new_angle)
        
        reward = math.cos(math.pi*(new_angle / 3)) - 0.5
        
        print "angle is ", new_angle
        print "reward is ",reward
        
        next_q = max(self.Q[self.next_idx])
        self.Q[self.idx][self.move] = self.Q[self.idx][self.move] + self.alpha * (reward + self.dis * next_q - self.Q[self.idx][self.move])
        self.idx = self.next_idx
        
    def torque_controller(self):
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            self.update_state()
            if self.e > 0.2:
                self.e = self.e * 0.99
            
            rate.sleep()
        for k,v in self.Q.items():
            print "angle ", k, "Q + ", v[1], "- ", v[0] 
if __name__ == '__main__':
    try:
        rl = RL()
        print "init done"
        rl.torque_controller()
    except rospy.ROSInterruptException:
        print "yay"
        pass
    
