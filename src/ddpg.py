import os
import math
import time
import random
import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Merge
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
import serial
import threading

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

class Weight(threading.Thread):
    def __init__(self, ser):
        super(Weight, self).__init__()
        self.ser = ser
        self.weight = 0
        
    def run(self):
        while True:
            try:
                weight = self.ser.readline().split(",")[1].split(" ")[0]
            except IndexError:
                continue
            try:
                self.weight = int(weight)
            except ValueError:
                continue
        

class RL(object):
    def __init__(self):
        rospy.init_node('dqn')
        self.pub2 = rospy.Publisher('/pan2_controller/command', Float64, queue_size=10)
        self.pub4 = rospy.Publisher('/pan4_controller/command', Float64, queue_size=10)
        self.pub5 = rospy.Publisher('/pan5_controller/command', Float64, queue_size=10)
        #self.init_pub = rospy.Publisher('/tilt4_controller/command', Float64, queue_size=10)
        rospy.Subscriber('/pan2_controller/state', JointState, self.joint2)
        rospy.Subscriber('/pan4_controller/state', JointState, self.joint4)
        rospy.Subscriber('/pan5_controller/state', JointState, self.joint5)
        COM = '/dev/ttyUSB2'
        ser = serial.Serial(COM, bytesize=serial.SEVENBITS, parity=serial.PARITY_EVEN, timeout=1.0)
        
        self.bad=False
        self.weight_thread = Weight(ser)
        self.weight_thread.daemon = True
        self.weight_thread.start()
        self.dxl_io = dynamixel_io.DynamixelIO('/dev/ttyUSB0',1000000)
        self.action_dim = 3
        self.sensor_dim = 3
        self.frame_num = 1
        self.img_dims = [84,84]
        self.initial_replay_size = 100
        self.gamma = 0.99
        self.epsilon = 1.0
        self.tau = 0.001
        self.final_epsilon = 0.1
        self.epsilon_step = 0.0001
        self.num_replay_memory = 50000
        self.target_update_interval = 1000
        self.momentum = 0.95
        self.min_grad = 0.01
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.t = 0

        self.freq = 4
        self.action_bound = 3.0
        self.rate = rospy.Rate(10)
        self.replay_memory = deque()
        self.weight = 0.0
        self.angle = np.array([0.0, 0.0, 0.0]).astype('float32')
        self.velocity = np.array([0.0, 0.0, 0.0]).astype('float32')
        
        self.video = Video(self.img_dims)
        
        self.actor_network, self.s_s_a, self.act = self.build_actor_network()        
        self.critic_network, self.s_s_c, self.s_a_c, self.q, self.action_grads = self.build_critic_network()

        self.target_actor_network, self.t_s_s_a, self.t_act = self.build_actor_network()
        
        self.target_critic_network, self.t_s_s_c, self.t_s_a_c, self.t_q, self.t_actor_grads = self.build_critic_network()
        
        
        self.actor_network_weights = self.actor_network.trainable_weights
        self.target_actor_network_weights = self.target_actor_network.trainable_weights
        self.update_target_actor_network = [self.update_target(self.target_actor_network_weights[i], self.actor_network_weights[i]) for i in xrange(len(self.target_actor_network_weights))]

        self.critic_network_weights = self.critic_network.trainable_weights
        self.target_critic_network_weights = self.target_critic_network.trainable_weights
        self.update_target_critic_network = [self.update_target(self.target_critic_network_weights[i], self.critic_network_weights[i]) for i in xrange(len(self.target_critic_network_weights))]
        self.batch_size = 32
        self.y, self.loss, self.grad_update, self.a, self.actor_optimize = self.build_training_op()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.update_target_actor_network)
        self.sess.run(self.update_target_critic_network)

    def joint2(self, msg):
        self.angle[0] = (msg.current_pos - 2.5) / 2.5
        self.velocity[0] = msg.current_pos

    def joint4(self, msg):
        self.angle[1] = (msg.current_pos - 2.5) / 2.5
        self.velocity[1] = msg.current_pos

    def joint5(self, msg):
        self.angle[2] = (msg.current_pos - 2.5) / 2.5
        self.velocity[2] = msg.current_pos

    def update_target(self, tar, original):
        tar = self.tau * original + (1-self.tau)*tar
        return tar

    def get_reward(self, sensor_state):
        #reward = -math.cos(math.pi*(self.angle * 0.333 + 0.1666))
        angle2 = sensor_state[0]
        angle4 = sensor_state[1]
        angle5 = sensor_state[2]
        print angle2, angle4, angle5

        reward = 0.0
        #reward for angle

        reward += -angle2
        if angle2 > 0.0:
            if angle4 < -0.2:
                reward -= 10.0
        weight = self.weight_thread.weight
        
        if weight > 500.0:
            reward -= 100.0
        elif weight > 1.0:
            reward += 10.0
        
        if weight < -1.0:
            reward += 500.0
        
        return reward
    
    def build_actor_network(self):
        final_model = Sequential()
        final_model.add(Dense(30, activation='relu', input_dim=self.sensor_dim))
        final_model.add(Dense(30, activation='relu'))

        final_model.add(Dense(self.action_dim, activation='tanh'))
        


        s_s_a = tf.placeholder(tf.float32, [None, self.sensor_dim])
        
        actions = tf.mul(final_model([s_s_a]), self.action_bound)

        return final_model, s_s_a, actions

    def build_critic_network(self):
        sensor_model = Sequential()
        sensor_model.add(Dense(30, activation='relu', input_dim=self.sensor_dim))
        #sensor_model.add(Dense(10, activation='tanh'))

        action_model = Sequential()
        action_model.add(Dense(30, activation='relu', input_dim=self.action_dim))
        #sensor_model.add(Dense(10, activation='tanh'))

        merged = Merge([sensor_model, action_model], mode='concat')

        final_model = Sequential()
        final_model.add(merged)
        #final_model.add(Dense(self.action_dim, activation='tanh'))
        final_model.add(Dense(self.action_dim, activation='linear'))

    
        s_s_c = tf.placeholder(tf.float32, [None, self.sensor_dim])
        s_a_c = tf.placeholder(tf.float32, [None, self.action_dim])

        q = final_model([s_s_c, s_a_c])

        action_grads = tf.gradients(q, s_a_c)

        return final_model, s_s_c, s_a_c, q, action_grads
        
    def get_image_state(self):
        state = self.video.get_state()
        state = np.float32(state.reshape((state.shape[0], state.shape[1], 1)))
        return state

    def get_sensor_state(self):
        return self.angle
        
    def get_action(self, sensor_state):
        if self.t % self.freq == 0:
            action = self.act.eval(feed_dict={self.s_s_a: [np.float32(sensor_state)]})[0]
            action += np.random.normal(scale=1.5, size=3)
        else:
            action = self.pred_action
        self.pred_action = action
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
        #speed2 = random.uniform(2.0, 3.5)
        #speed4 = random.uniform(1.0, 2.5)
        speed5 = random.uniform(1.0, 2.5)

        self.pub2.publish(4.0)
        self.pub4.publish(2.5)
        if r > 0.5:
            #self.pub4.publish(speed4)
            self.pub5.publish(speed5)
        else:
            #self.pub4.publish(-speed4)
            self.pub5.publish(-speed5)
        print "init_arm"
        time.sleep(2.5)

    def init_arm_first(self):
        r = random.random()
        speed2 = random.uniform(2.5, 4.0)
        speed4 = random.uniform(1.0, 2.5)
        speed5 = random.uniform(1.0, 2.5)

        if r > 0.5:
            self.pub2.publish(speed2)
            self.pub4.publish(speed4)
            self.pub5.publish(speed5)
        else:
            self.pub2.publish(-speed2)
            self.pub4.publish(-speed4)
            self.pub5.publish(-speed5)
        print "init_arm"
        time.sleep(2.5)
        
    def build_training_op(self):
        a = tf.placeholder(tf.float32, [None, self.action_dim])
        y = tf.placeholder(tf.float32, [None, self.action_dim])
        
        
        #q_value = tf.reduce_sum(tf.mul(self.q, a_one_hot), 1)
        critic_error = tf.square(y-self.q)
        critic_loss = tf.reduce_mean(critic_error)
        #quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        #linear_part = error - quadratic_part
        #loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
        
        optimizer = tf.train.AdamOptimizer(self.critic_lr)
        grad_update = optimizer.minimize(critic_loss, var_list=self.critic_network_weights)
        
        params_grad = tf.gradients(self.act, self.actor_network_weights, -a)
        grads = zip(params_grad, self.actor_network_weights)
        actor_optimize = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(grads)
        
        
        #grad_update = optimizer.minimize(loss, var_list=self.actor_network_weights + self.critic_network_weights)

        return y,critic_loss,grad_update,a,actor_optimize

    def cut(self, actions):
        for n,action in enumerate(actions):
            if n == 0:
                if action < -3.0:
                    actions[n] = -3.0
                elif action > 3.0:
                    actions[n] = 3.0
            else:
                if action < -1.5:
                    actions[n] = -1.5
                elif action > 1.5:
                    actions[n] = 1.5
        return actions
    
    def run(self):
        t = 0
        r_mean = 0
        if self.t > self.initial_replay_size:
            self.init_arm_first()
        else:
            self.init_arm_first()
        termination = False
        next_sensor_state = self.get_sensor_state()
        while not termination:
            sensor_state = next_sensor_state
            action = self.get_action(sensor_state)

            action = self.cut(action)
            
            self.pub2.publish(action[0])
            self.pub4.publish(action[1])
            self.pub5.publish(action[2])
            
            self.rate.sleep()
            
            next_sensor_state = self.get_sensor_state()
            
            reward = self.get_reward(next_sensor_state)
            r_mean += reward
            self.learn(sensor_state,action,reward,next_sensor_state)

            #self.check_state()
            
            t += 1
            self.t += 1
            print "Time:%d Action:%s Reward:%f" % (self.t, action, reward)
            if t == 100:
                print r_mean
                termination = True
            if self.bad == True:
                print r_mean
                termination = True
                
    def learn(self, sensor_state, action, reward, next_sensor_state):
        self.replay_memory.append((sensor_state, action, reward, next_sensor_state))

        if len(self.replay_memory) > self.num_replay_memory:
            self.replay_memory.popleft()

        if self.t > self.initial_replay_size:
            self.train()
            
        self.sess.run(self.update_target_actor_network)
        self.sess.run(self.update_target_critic_network)
            
    def train(self):
        sensor_state_batch = []
        action_batch = []
        reward_batch = []
        next_sensor_state_batch = []
        y_batch = []

        minibatch = random.sample(self.replay_memory, self.batch_size)
        for data in minibatch:
            sensor_state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_sensor_state_batch.append(data[3])
                                     
        
        #generate target signals
        target_action_batch = self.t_act.eval(feed_dict={
            self.t_s_s_a: np.float32(np.array(next_sensor_state_batch))
            })
        
        target_q_values_batch = self.t_q.eval(feed_dict={
            self.t_s_s_c: np.float32(np.array(next_sensor_state_batch)),
            self.t_s_a_c: np.float32(np.array(target_action_batch))})
            
        y_batch = np.float32(np.array(reward_batch)[:,None] + self.gamma * target_q_values_batch)

        sampled_actions = self.act.eval(feed_dict={
            self.s_s_a: np.float32(np.array(sensor_state_batch))
            })
        
        critic_loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.y: y_batch,
            self.s_s_c: np.float32(np.array(sensor_state_batch)),
            self.s_a_c: np.float32(np.array(action_batch))
        })

        action_grads = self.action_grads[0].eval(feed_dict={
            self.s_s_c: np.float32(np.array(sensor_state_batch)),
            self.s_a_c: np.float32(np.array(sampled_actions)),
            })
        
        self.sess.run(self.actor_optimize, feed_dict={
            self.a: np.float32(np.array(action_grads)),
            self.s_s_a: np.float32(np.array(sensor_state_batch))
            #self.s_a_c: np.float32(np.array(action_batch))
        })
            
    def main(self):
        max_episode = 10000
        count = 0
        os.system('rosrun dynamixel_driver set_servo_config.py 2 --cw-angle-limit=0 --ccw-angle-limit=0')
        os.system('rosrun dynamixel_driver set_servo_config.py 4 --cw-angle-limit=0 --ccw-angle-limit=0')
        os.system('rosrun dynamixel_driver set_servo_config.py 5 --cw-angle-limit=0 --ccw-angle-limit=0')
        while not rospy.is_shutdown():
            self.run()
        
if __name__ == '__main__':
    rl = RL()
    try:
        rl.main()
    except rospy.ROSInterruptException:
        pass
        
