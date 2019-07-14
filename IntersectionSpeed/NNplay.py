import tensorflow as tf
import numpy as np
import random
import time
import copy

from collections import deque
from IntersectionDirection import Intersection
from GUI import GUI
from PGNetwork import PGNetwork
from DQNetwork import DQNetwork
from DQNetwork import Memory



numRoads = 1
length = 3  
prob = 0.1
speed = 1
keepalive = -2
wait_weight = 0.7

state_size = 20
action_size = 3
learning_rate = 0.000002

total_episodes = 2000
max_steps = 300
batch_size = 500
epoch = 0
num_epochs = 100

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00015

gamma = 0.95

memory_size = 10000
pretrain_length = memory_size
grid_space = 50


def create_environment():
	game = Intersection(numRoads = numRoads, length = length, prob = prob, speed = speed, keepalive = keepalive, wait_weight = wait_weight)

	idle = [1,0,0]
	NSGo = [0,1,0]
	EWGo = [0,0,1]
	possible_actions = [idle, NSGo, EWGo]

	return game, possible_actions


with tf.Session() as sess:

	PGNetwork = PGNetwork( 
						  state_shape=state_size, 
						  action_shape = action_size, 
						  learning_rate=learning_rate,
						  name = "PGNetwork")
	game, possible_actions = create_environment()
	totalScore = 0
	saver = tf.train.Saver()

	saver.restore(sess, "./PGModels_1speed_1x1(3)_0.5/model.ckpt")
	window = GUI(grid_space)

	step = 0

	for i in range(10):

		game.newInstance()
		while not game.gameEnd():
			step += 1
			if(step == 30):
				print("TOTAL_SCORE", totalScore)
				step = 0
			frame = game.getSimpleState()
			action_probability_distribution = sess.run(PGNetwork.action_distribution, feed_dict={PGNetwork.inputs_: frame.reshape(1, state_size)})
			#print(game.gameWait())
			#action = np.random.choice(range(action_probability_distribution.shape[1]), p = action_probability_distribution.ravel())
			action = np.argmax(action_probability_distribution)
			#print(action)
			action = possible_actions[action]
			'''Qs = sess.run(DQNetwork.outputs, feed_dict={DQNetwork.inputs_: frame.reshape(1, state_size)})
			action = np.argmax(Qs)
			action = possible_actions[int(action)]'''
			#reward = game.step(NS_action=action[1], EW_action=action[2])
			reward = game.step(action[1], action[2])
			window.update(game.binaryRepresentation())
			print("Score: ", reward)
			totalScore += reward
			time.sleep(0.5)
		print("GAME END, TOTAL_SCORE:", totalScore)

