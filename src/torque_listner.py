#!/usr/bin/env python
import rospy
from dynamixel_msgs.msg import JointState
import sys

def callback(data):
    rospy.loginfo("Joint 3's state is now %s", data.current_pos)
    
def listener():
    rospy.init_node('torque_listener', anonymous=True)
    rospy.Subscriber('/pan_controller/state', JointState, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
