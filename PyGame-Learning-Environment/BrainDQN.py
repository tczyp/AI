# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

import os
import tensorflow as tf
import numpy as np
import random
from collections import deque
import cv2

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100.  # timesteps to observe before training
EXPLORE = 150000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  #
#  starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 128  # size of minibatch
RL = 1e-7

class BrainDQN:
    def __init__(self, action_set, map_width, map_height):
        self.no_random = False
        self.game_turn = 0
        self.game_step = 0
        self.max_step = 0
        self.cur_turn_steps = []
        self.ac_tde = 2
        # init replay memory
        self.replayMemory = deque()

        self.map_width = map_width
        self.map_height = map_height

        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.action_set = action_set
        self.actions = len(action_set)
        # self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        # init Q network
        self.createQNetwork()

    def createQNetwork(self):
        with tf.device("/CPU:0"):
            # network weights
            W_conv1 = self.weight_variable([8, 8, 4, 32])
            b_conv1 = self.bias_variable([32])

            W_conv2 = self.weight_variable([4, 4, 32, 64])
            b_conv2 = self.bias_variable([64])

            W_conv3 = self.weight_variable([3, 3, 64, 64])
            b_conv3 = self.bias_variable([64])

            neural_count = 2048

            # W_fc1 = self.weight_variable([1600, 512])
            b_fc1 = self.bias_variable([neural_count])

            W_fc2 = self.weight_variable([neural_count, self.actions])
            b_fc2 = self.bias_variable([self.actions])

            # input layer

            self.stateInput = tf.placeholder("float", [None, self.map_width, self.map_height, 4])
            self.tf_acts = tf.placeholder(tf.float32, [None, self.actions], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

            # hidden layers
            h_conv1 = tf.nn.relu(self.conv2d(self.stateInput, W_conv1, 4) + b_conv1)
            h_pool1 = self.max_pool_2x2(h_conv1)

            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)

            h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)

            # h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
            # h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

            cur_shape = h_conv3.get_shape()
            d = int(cur_shape[1] * cur_shape[2] * cur_shape[3])
            h_conv3_flat = tf.reshape(h_conv3, [-1, d])
            W_fc1 = self.weight_variable([d, neural_count])
            hide_layer = tf.nn.tanh(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
            self.all_act = tf.matmul(hide_layer, W_fc2) + b_fc2
            self.QValue = self.all_act
            self.all_act_prob = tf.nn.softmax(self.all_act)

            # self.QValue = self.all_act_prob
            # Q Value layer
            # self.QValue = tf.matmul(h_fc1, W_fc2) + b_fc2

            # self.actionInput = tf.placeholder("float", [None, self.actions])

            # self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.all_act, labels=self.tf_acts)
            # self.loss = tf.reduce_mean(self.neg_log_prob * self.tf_vt)
            # self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.loss)

            # self.yInput = tf.placeholder("float", [None])
            Q_action = tf.reduce_sum(tf.multiply(self.QValue, self.tf_acts), reduction_indices=1)
            self.loss = tf.reduce_mean(tf.square(self.tf_vt - Q_action))
            self.trainStep = tf.train.AdamOptimizer(RL).minimize(self.loss)

            # saving and loading networks
            self.saver = tf.train.Saver()
            config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            # config.log_device_placement = True
            config.allow_soft_placement = True
            # config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
            self.session = tf.Session(config=config)
            # self.session = tf.InteractiveSession()
            # self.session.run(tf.initialize_all_variables())
            self.session.run(tf.global_variables_initializer())

            checkpoint = tf.train.get_checkpoint_state("saved_networks")
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                print("Could not find old network weights")

    # def trainQNetwork(self):
    #     # Step 1: obtain random minibatch from replay memory
    #     minibatch = random.sample(self.replayMemory, BATCH_SIZE)
    #     state_batch = [data[0] for data in minibatch]
    #     action_batch = [data[1] for data in minibatch]
    #     reward_batch = [data[2] for data in minibatch]
    #     nextState_batch = [data[3] for data in minibatch]
    #
    #     # Step 2: calculate y
    #     y_batch = []
    #     QValue_batch = self.QValue.eval(feed_dict={self.stateInput: nextState_batch})
    #     for i in range(0, BATCH_SIZE):
    #         terminal = minibatch[i][4]
    #         if terminal:
    #             y_batch.append(reward_batch[i])
    #         else:
    #             y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))
    #
    #     self.trainStep.run(feed_dict={
    #         self.yInput: y_batch,
    #         self.actionInput: action_batch,
    #         self.stateInput: state_batch
    #     })
    #
    #     # save network every 100000 iteration
    #     if self.timeStep % 10000 == 0:
    #         self.saver.save(self.session, 'saved_networks_my_new/' + 'network' + '-dqn', global_step=self.timeStep)

    def setPerception(self, nextObservation, action, reward, terminal):
        with tf.device("/CPU:0"):
            # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)

            nextObservation = np.reshape(nextObservation, (self.map_width, self.map_height, 1))
            newState = np.append(self.currentState[:, :, 1:], nextObservation, axis=2)
            cv2.imshow('xxx', newState)
            #self.replayMemory.append((self.currentState, action, reward, newState, terminal))
            # replay_item = (self.currentState, action, reward, newState, terminal)
            # self.good_steps.append(replay_item)

            self.store_transition(self.currentState, action, reward)

            if terminal:
                self.game_turn += 1
                self.learn()

            # if len(self.replayMemory) > REPLAY_MEMORY:
            #     self.replayMemory.popleft()
            # if self.timeStep > OBSERVE:
            #     # Train the network
            #     self.trainQNetwork()

            self.currentState = newState
            self.timeStep += 1

            # save network every 100000 iteration
            if self.timeStep % 5000 == 0:
                if not os.path.exists('saved_networks'):
                    os.mkdir('saved_networks')
                self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step=self.timeStep)

    def pickAction(self, reward, obs):
        return self.action_set[np.random.randint(0, len(self.action_set))]

    def getAction(self):
        # QValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentState]})[0]
        action = np.zeros(self.actions)
        # action_index = 0
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
                # prob_weights = self.session.run(self.all_act_prob, feed_dict={self.stateInput: [self.currentState]})
                # action_index = np.random.choice(range(prob_weights.shape[1]),
                #                                 p=prob_weights.ravel())  # select action w.r.t the actions prob
                # action[action_index] = 1
            else:
                prob_weights = self.session.run(self.all_act_prob, feed_dict={self.stateInput: [self.currentState]})
                # action_index = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
                action_index = np.argmax(prob_weights)
                action[action_index] = 1
        else:
            action_index = 5
            action[action_index] = 1  # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return self.action_set[action_index], action

    def setInitState(self, observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis=2)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def learn(self):
        # discount and normalize episode reward
        with tf.device("/CPU:0"):
            self._discount_and_norm_rewards()

            # train on episode
            s_list,a_list,q_list = list(zip(*self.cur_turn_steps))
            _,loss,all_act_prob,all_act,tf_acts = self.session.run([self.trainStep, self.loss, self.all_act_prob, self.all_act, self.tf_acts], feed_dict={
                self.stateInput: s_list,  # shape=[None, n_obs]
                self.tf_acts: np.array(a_list),  # shape=[None, ]
                self.tf_vt: np.array(q_list),  # shape=[None, ]
            })

            if len(self.replayMemory) > BATCH_SIZE * 4:
                for k in range(10):
                    minibatch = random.sample(self.replayMemory, BATCH_SIZE)
                    s_list, a_list, q_list = list(zip(*minibatch))

                    _, loss, all_act_prob, all_act, tf_acts = self.session.run(
                        [self.trainStep, self.loss, self.all_act_prob, self.all_act, self.tf_acts], feed_dict={
                        self.stateInput: s_list,  # shape=[None, n_obs]
                        self.tf_acts: np.array(a_list),  # shape=[None, ]
                        self.tf_vt: np.array(q_list),  # shape=[None, ]
                    })

            for item in self.cur_turn_steps:
                self.replayMemory.append(item)

            re_len = len(self.replayMemory)
            print ('re_len', re_len)
            while re_len > REPLAY_MEMORY:
                self.replayMemory.popleft()
                re_len = len(self.replayMemory)

            # self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
            self.cur_turn_steps = []
            # return discounted_ep_rs_norm

    def store_transition(self, s, a, r):
        self.cur_turn_steps.append([s, a, r])
        # self.ep_obs.append(s)
        # # for i,v in enumerate(a):
        # #     if v == 1:
        # #         self.ep_as.append(i)
        # self.ep_as.append(a)
        # self.ep_rs.append(r)

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        # discounted_ep_rs = np.zeros_like(self.ep_rs)
        # discounted_ep_rs = [0] * len(self.cur_turn_steps)
        running_add = 0

        # for t in reversed(range(0, len(self.ep_rs))):
        addition_list = []
        total_tde = 0
        for i in reversed(range(len(self.cur_turn_steps))):
            # # action_index = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
            # action_index = np.argmax(prob_weights)
            item = self.cur_turn_steps[i]
            reward = item[2]
            q_values = self.session.run(self.QValue, feed_dict={self.stateInput: [item[0]]})
            eval_max_q = np.max(q_values)
            if i == len(self.cur_turn_steps) - 1:
                new_q = reward
                e_q_value = reward
                tde = abs(new_q - eval_max_q)
            else:
                new_q = e_q_value * GAMMA + reward
                ratio = max(self.ac_tde, 0)
                ratio = min(ratio, 1)
                e_q_value = max(new_q, new_q * ratio + eval_max_q * (1 - ratio))
                tde = abs(new_q - eval_max_q)

            print(reward, new_q, eval_max_q, e_q_value, tde)
            total_tde += tde ** 2
            item[2] = e_q_value

            # if tde is big, take more changes to improve it
            bi = len(self.cur_turn_steps) - i
            if bi < 10 and True:
                for _ in range(max(int(5 / bi), 3)):
                    if tde > random.random():
                        addition_list.append(item)

        total_tde = np.sqrt(total_tde / len(self.cur_turn_steps))

        self.ac_tde = self.ac_tde * 0.9 + total_tde * 0.1

        # normalize episode rewards
        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        print('cur_turn_steps', len(self.cur_turn_steps))
        print('total_tde', total_tde)
        # print ('discounted_ep_rs', len(discounted_ep_rs))
        self.cur_turn_steps.extend(addition_list)
        print('cur_turn_steps addition_list', len(self.cur_turn_steps))

        # return discounted_ep_rs