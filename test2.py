import tensorflow as tf
import numpy as np
import random
import time
import copy

from collections import deque
from Intersection import Intersection
from GUI import GUI
from NeuralNet import NeuralNet
from NeuralNet import Memory


length = 5
prob = 0.5
keepalive = -2
wait_weight = 10

state_size = 2*(length + 2)
action_size = 3
learning_rate = 0.0002

total_episodes = 500
max_steps = 300
batch_size = 1000
num_epoch = 20

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

gamma = 0.99

pretrain_length = batch_size
memory_size = 5000000
training = True

grid_space = 50


def create_environment():
	game = Intersection(length=length, prob = prob, keepalive = keepalive, wait_weight=wait_weight)

	idle = [1,0,0]
	waitNS = [0,1,0]
	waitEW = [0,0,1]
	possible_actions = [idle, waitNS, waitEW]

	return game, possible_actions


with tf.Session() as sess:

	DQNetwork = NeuralNet(batch_size=batch_size, 
						  state_shape=state_size, 
						  action_shape = action_size, 
						  learning_rate=learning_rate)
	game, possible_actions = create_environment()
	totalScore = 0
	saver = tf.train.Saver()

	saver.restore(sess, "./models/model.ckpt")
	window = GUI(50)

	for i in range(10):

		game.newInstance()
		while not game.gameEnd():
			frame = game.getState()
			Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: frame.reshape(1, state_size)})
			action = np.argmax(Qs)
			action = possible_actions[int(action)]
			reward = game.step(NS_action=action[1], EW_action=action[2])
			window.update(frame)
			print("Score: ", reward)
			totalScore += reward
		print("TOTAL_SCORE", totalScore)

