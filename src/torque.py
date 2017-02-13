#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
from dynamixel_msgs.msg import JointState
import sys

def callback(data):
    rospy.loginfo("State %s", data.current_pos)

def torque_move(pub, speed, rate):
    pub.publish(speed)
    rospy.loginfo(speed)
    rate.sleep()
    pub.publish(0.0)
    rate.sleep()

def torque_controller():
    pub = rospy.Publisher('/pan_controller/command', Float64, queue_size = 10)
    #rospy.Subscriber('/pan_controller/state', JointState, callback)
    rospy.init_node('torque_controller', anonymous=True)
    #rospy.init_node('listner', anonymous=True)
    rate = rospy.Rate(0.4)
    speed = 2.0
    speed = 0.0
    while not rospy.is_shutdown():
        torque_move(pub, speed, rate)
        speed = -speed

if __name__ == '__main__':
    try:
        torque_controller()
    except rospy.ROSInterruptException:
        pass
