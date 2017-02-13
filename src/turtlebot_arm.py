#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64
from dynamixel_msgs.msg import JointState
import sys

def move(pubs, states, rate):
    for pub, state in zip(pubs, states):
        pub.publish(state)
        rospy.loginfo(state)
    rate.sleep()



def commander():
    pub1 = rospy.Publisher('/tilt1_controller/command', Float64, queue_size=10)
    pub2 = rospy.Publisher('/tilt2_controller/command', Float64, queue_size=10)
    pub3 = rospy.Publisher('/tilt3_controller/command', Float64, queue_size=10)
    pub4 = rospy.Publisher('/tilt4_controller/command', Float64, queue_size=10)
    pub5 = rospy.Publisher('/tilt5_controller/command', Float64, queue_size=10)
    pubs = [pub1,pub2,pub3,pub4,pub5]

    rospy.init_node('commander', anonymous=True)
    
    rate = rospy.Rate(0.5)
    command = 0.5
    init_states=[0.0, 0.0, 0.0, 0.0, 0.0]
    move(pubs, init_states, rate)
    #sys.exit()
    while not rospy.is_shutdown():
        first = [1.5, 0.0, 0.0, 0.0, 0.0]
        move(pubs, first, rate)
        
        second = [1.5, 1.05, -1.0, -1.0, -0.5]
        move(pubs, second, rate)

        third = [1.5, 1.05, -1.0, -1.0, 0.2]
        move(pubs, third, rate)

        fourth = [1.5, 0.0, 0.0, 0.0, 0.2]
        move(pubs, fourth, rate)

        fifth = [0.0, 0.0, 0.0, 0.0, 0.2]
        move(pubs, fifth, rate)

        sixth = [0.0, 0.0, -1.5, -1.5, 0.2]
        move(pubs, sixth, rate)

        last = [0.0, 0.0, -1.5, -1.5, -0.5]
        move(pubs, last, rate)

        move(pubs, init_states, rate)
        

if __name__ == '__main__':
    try:
        commander()
    except rospy.ROSInterruptException:
        pass
