#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64
from dynamixel_msgs.msg import JointState
import sys
import os

def move(pubs, states, rate, motor_ids):
    for pub, state, motor_id in zip(pubs, states, motor_ids):
        os.system('rosrun dynamixel_driver set_servo_config.py %d --cw-angle-limit=0 --ccw-angle-limit=1023' % motor_id)
        pub.publish(state)
        rospy.loginfo(state)
    rate.sleep()



def commander():
    
    pub1 = rospy.Publisher('/tilt1_controller/command', Float64, queue_size=10)
    pub2 = rospy.Publisher('/tilt2_controller/command', Float64, queue_size=10)
    pub3 = rospy.Publisher('/tilt3_controller/command', Float64, queue_size=10)
    pub4 = rospy.Publisher('/tilt4_controller/command', Float64, queue_size=10)
    pub5 = rospy.Publisher('/tilt5_controller/command', Float64, queue_size=10)
    pubs = [pub1,pub2,pub3,pub5]
    motor_ids = [1,2,3,5]

    rospy.init_node('commander', anonymous=True)
    
    rate = rospy.Rate(1)
    command = 0.5
    init_states=[0.0 for i in pubs]
    while not rospy.is_shutdown():
        move(pubs, init_states, rate, motor_ids)
        
        

if __name__ == '__main__':
    try:
        commander()
    except rospy.ROSInterruptException:
        pass
