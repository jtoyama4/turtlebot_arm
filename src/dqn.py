import os
import math
import time
import random
import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D as Conv2d
import cv2
import sys
from dynamixel_msgs.msg import JointState
from std_msgs.msg import Float64
from dynamixel_msgs.msg import JointState
import rospy
import math
import tensorflow as tf
from std_msgs.msg import Float64
from collections import deque
from dynamixel_driver import dynamixel_io

class Video(object):
    def __init__(self,img_dims):
        self.cap = cv2.VideoCapture(0)
        self.img_dims = img_dims
        
    def get_state(self):
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(self.img_dims[0], self.img_dims[1]))
        gray = gray / 255.0
        return gray

class RL(object):
    def __init__(self):
        rospy.init_node('dqn')
        self.pub = rospy.Publisher('/pan4_controller/command', Float64, queue_size=10)
        self.init_pub = rospy.Publisher('/tilt4_controller/command', Float64, queue_size=10)
        rospy.Subscriber('/pan4_controller/state', JointState, self.angle_state)
        self.dxl_io = dynamixel_io.DynamixelIO('/dev/ttyUSB0',1000000)
        self.num_actions = 2
        self.frame_num = 1
        self.img_dims = [84,84]
        self.initial_replay_size = 100
        self.gamma = 0.99
        self.epsilon = 1.0
        self.final_epsilon = 0.1
        self.epsilon_step = 0.0002
        self.num_replay_memory = 40000
        self.target_update_interval = 1000
        self.momentum = 0.95
        self.min_grad = 0.01
        self.lr = 0.00025
        self.t = 0
        self.rate = rospy.Rate(5)
        self.replay_memory = deque()
        
        self.video = Video(self.img_dims)
        
        self.q_network, self.s, self.q = self.build_network()
        self.target_network, self.st, self.qt = self.build_network()
        
        self.q_network_weights = self.q_network.trainable_weights
        self.target_network_weights = self.target_network.trainable_weights
        self.update_target_network = [self.target_network_weights[i].assign(self.q_network_weights[i]) for i in xrange(len(self.target_network_weights))]
        self.batch_size = 32
        self.a, self.y, self.loss, self.grad_update = self.build_training_op()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.update_target_network)
    def angle_state(self, msg):
        self.angle = msg.current_pos

    def get_reward(self):
        reward = -math.cos(math.pi*(self.angle * 0.333 + 0.1666))
        return reward
    
    def build_network(self):
        model = Sequential()
        model.add(Conv2d(32, 8, 8, input_shape=(self.img_dims[0], self.img_dims[1], self.frame_num), activation='relu', subsample=(4,4)))
        model.add(Conv2d(64,4,4, activation='relu', subsample=(2,2)))
        model.add(Conv2d(64,3,3, activation='relu', subsample=(1,1)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        s = tf.placeholder(tf.float32, [None, self.img_dims[0], self.img_dims[1], self.frame_num])
        q = model(s)

        return model,s,q
    
    def get_state(self):
        state = self.video.get_state()
        state = state.reshape((state.shape[0], state.shape[1], 1))
        return state
        
    def get_action(self, state):
        if self.epsilon >= random.random() or self.t < self.initial_replay_size:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q.eval(feed_dict={self.s: [np.float32(state)]}))

        if self.epsilon > self.final_epsilon and self.t >= self.initial_replay_size:
            self.epsilon -= self.epsilon_step
        return action
    
    """def init_arm(self):
        #os.system('rosrun dynamixel_driver set_servo_config.py 4 --cw-angle-limit=0 --ccw-angle-limit=1023')
        self.dxl_io.set_angle_limit_ccw(4, 1023)
        self.rate.sleep()
        init_angle = random.uniform(-1.5,1.5)
        self.init_pub.publish(init_angle)
        self.angle = init_angle
        self.rate.sleep()
        #os.system('rosrun dynamixel_driver set_servo_config.py 4 --cw-angle-limit=0 --ccw-angle-limit=0')
        self.dxl_io.set_angle_limit_ccw(4, 0)
        self.rate.sleep()
    """

    def init_arm(self):
        r = random.random()
        speed = random.uniform(1.0, 2.5)
        if r > 0.5:
            self.pub.publish(speed)
            d = 1
        else:
            self.pub.publish(-speed)
            d = 0
        print "init_arm Direction:%d speed:%f" % (d, speed)
        time.sleep(1.5)
        
    def build_training_op(self):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])
        
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.mul(self.q, a_one_hot), 1)

        error = tf.abs(y-q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(self.lr, momentum=self.momentum, epsilon=self.epsilon)
        grad_update = optimizer.minimize(loss, var_list=self.q_network_weights)

        return a,y,loss,grad_update
                                    
    def run(self):
        t = 0
        r_mean = 0
        self.init_arm()
        termination = False
        next_state = self.get_state()
        while not termination:
            state = next_state
            action = self.get_action(state)
            if action == 0:
                speed = -1.0
            else:
                speed = 1.0
            self.pub.publish(speed)
            self.rate.sleep()
            next_state = self.get_state()
            reward = self.get_reward()
            r_mean += reward
            self.learn(state,action,reward,next_state)
            
            t += 1
            self.t += 1
            print "Time:%d Action:%s Reward:%f" % (self.t, action, reward)
            if t == 50:
                print r_mean
                termination = True
                
    def learn(self, state, action ,reward ,next_state):
        self.replay_memory.append((state, action, reward, next_state))

        if len(self.replay_memory) > self.num_replay_memory:
            self.replay_memory.popleft()

        if self.t > self.initial_replay_size:
            self.train()
        if self.t % self.target_update_interval == 0:
            self.sess.run(self.update_target_network)
            
    def train(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        y_batch = []

        minibatch = random.sample(self.replay_memory, self.batch_size)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
        
        

        #generate target signals
        target_q_values_batch = self.qt.eval(feed_dict={self.st: np.float32(np.array(next_state_batch))})
        y_batch = reward_batch + self.gamma * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.s: np.float32(np.array(state_batch)),
            self.a: action_batch,
            self.y: y_batch
        })
    def main(self):
        max_episode = 10000
        count = 0
        os.system('rosrun dynamixel_driver set_servo_config.py 4 --cw-angle-limit=0 --ccw-angle-limit=0')
        while not rospy.is_shutdown():
            self.run()
        
if __name__ == '__main__':
    rl = RL()
    try:
        rl.main()
    except rospy.ROSInterruptException:
        pass
        
