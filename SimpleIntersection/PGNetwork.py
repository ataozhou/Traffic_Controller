import tensorflow as tf
import numpy as np
import random
import time
from collections import deque


class PGNetwork:

	def __init__(self, state_shape, action_shape, learning_rate, name = 'PGNetwork'):
		self.state_shape = state_shape
		self.action_shape = action_shape
		self.learning_rate = learning_rate
		self.n_nodes_l1 = int((((2* state_shape) + action_shape) / 3 + 4))
		self.n_nodes_l2 = int(((state_shape + (2 * action_shape)) / 3 + 4))

		with tf.variable_scope(name):
			self.inputs_ = tf.placeholder(tf.float32, [None, int(state_shape)], name="inputs_")
			self.actions = tf.placeholder(tf.float32, [None, int(action_shape)], name="actions")
			self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards_")

			self.layer1 = {'weights':tf.Variable(tf.random_normal([self.state_shape, self.n_nodes_l1])), 'biases':tf.Variable(tf.random_normal([self.n_nodes_l1]))}
			self.layer2 = {'weights':tf.Variable(tf.random_normal([self.n_nodes_l1, self.n_nodes_l2]) * np.sqrt(2 / self.n_nodes_l1)), 'biases':tf.Variable(tf.random_normal([self.n_nodes_l2]))}
			self.output = {'weights':tf.Variable(tf.random_normal([self.n_nodes_l2, self.action_shape]) * np.sqrt(2 / self.n_nodes_l2)), 'biases':tf.Variable(tf.random_normal([self.action_shape]))}

			self.l1 = tf.add(tf.matmul(self.inputs_, self.layer1['weights']), self.layer1['biases'])
			self.l1 = tf.nn.relu(self.l1)

			self.l2 = tf.add(tf.matmul(self.l1, self.layer2['weights']), self.layer2['biases'])
			self.l2 = tf.nn.relu(self.l2)

			self.outputs = tf.add(tf.matmul(self.l2, self.output['weights']), self.output['biases'])

			self.action_distribution = tf.nn.softmax(self.outputs)
			self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.outputs, labels = self.actions)
			self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_)
			self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

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


