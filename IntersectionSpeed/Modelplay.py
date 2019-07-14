import tensorflow as tf
import numpy as np
import time
import copy

from collections import deque
from IntersectionRoads import Intersection
from GUI import GUI
from PGNetwork import PGNetwork
from DQNetwork import DQNetwork
from DQNetwork import Memory



numRoads = 2
length = 5  
prob = 0.3
speed = 1
keepalive = -2
wait_weight = 0.7

state_size = 40
action_size = 4
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
	game = Intersection(numRoads = numRoads, length = length, prob = prob, speed = speed, wait_weight = wait_weight)

	FirstStreet = [1,0,0,0]
	SecondStreet = [0,1,0,0]
	ThirdStreet = [0,0,1,0]
	FourthStreet = [0,0,0,1]
	idle = [0,0,0,0,1]
	possible_actions = [FirstStreet, SecondStreet, ThirdStreet, FourthStreet, idle]

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

	saver.restore(sess, "./PGModels_1speed_2x2/model.ckpt")
	window = GUI(grid_space)

	step = 0

	for i in range(10):

		game.newInstance()
		step = 0
		while (not game.gameEnd()) and (step < 4500):
			step += 1
			'''print(game.NSRoads[0].numCars)
			print(game.EWRoads[0].numCars)
			if(step == 1000):
				print("TOTAL_SCORE", totalScore)
				print(game.NSRoads[0].prob)
				print(gam.EWRoads[0].prob)
				step = 0'''
			#action = game.chooseRoad3()
			#reward = game.step(1, 0)
			frame = game.getSimpleState()
			action_probability_distribution = sess.run(PGNetwork.action_distribution, feed_dict={PGNetwork.inputs_: frame.reshape(1, state_size)})
			action = np.argmax(action_probability_distribution)
			action = possible_actions[action]
			reward = game.step(np.argmax(action))
			text = raw_input()
			print(game.binaryRepresentation())
			window.update(game.binaryRepresentation())
			print("Score: ", reward)
			print("Total Wait: ", game.getWait())
			totalScore += reward
		print("GAME END, TOTAL_SCORE:", totalScore)

