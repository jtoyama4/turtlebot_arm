#!/usr/bin/env python
import rospy
from dynamixel_msgs.msg import JointState
import sys

def callback(data):
    rospy.loginfo("Joint 3's state is now %s", data.goal_pos)
    
def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/tilt3_controller/state', JointState, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
