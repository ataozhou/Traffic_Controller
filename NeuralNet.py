import tensorflow as tf
import numpy as np
import random
import time
from collections import deque


class NeuralNet:

	def __init__(self, batch_size, state_shape, action_shape, learning_rate, name = 'DQNetwork'):
		self.state_shape = int(state_shape)
		self.action_shape = int(action_shape)
		self.learning_rate = learning_rate
		self.n_nodes_l1 = int((((2* state_shape) + action_shape) / 3 + 4))
		self.n_nodes_l2 = int(((state_shape + (2 * action_shape)) / 3 + 4))

		with tf.variable_scope(name):
			self.inputs_ = tf.placeholder(tf.float32, [None, *state_shape])
			self.actions_ = tf.placeholder(tf.float32, [None, self.action_shape])
			self.target_Q = tf.placeholder(tf.float32, [None])

			self.layer1 = {'weights':tf.Variable(tf.random_normal([self.state_shape, self.n_nodes_l1])), 'biases':tf.Variable(tf.random_normal([self.n_nodes_l1]))}
			self.layer2 = {'weights':tf.Variable(tf.random_normal([self.n_nodes_l1, self.n_nodes_l2]) * np.sqrt(2 / self.n_nodes_l1)), 'biases':tf.Variable(tf.random_normal([self.n_nodes_l2]))}
			self.output = {'weights':tf.Variable(tf.random_normal([self.n_nodes_l2, self.action_shape]) * np.sqrt(2 / self.n_nodes_l2)), 'biases':tf.Variable(tf.random_normal([self.action_shape]))}

			self.l1 = tf.add(tf.matmul(self.inputs_, self.layer1['weights']), self.layer1['biases'])
			self.l1 = tf.nn.relu(self.l1)

			self.l2 = tf.add(tf.matmul(self.l1, self.layer2['weights']), self.layer2['biases'])
			self.l2 = tf.nn.relu(self.l2)

			self.output = tf.add(tf.matmul(self.l2, self.output['weights']), self.output['biases'])

			self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis = 1)
			self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
			self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

class Memory():

	def __init__(self, max_size):
		self.buffer = deque(maxlen = max_size)
		self.batchReference = 0

	def add(self, experience):
		self.buffer.append(experience)

	def sample(self, batch_size):
		buffer_size = len(self.buffer)
		index = np.random.choice(np.arange(buffer_size), size = batch_size, replace = False)
		return [self.buffer[i] for i in index]


