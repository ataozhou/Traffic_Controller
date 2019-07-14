import tensorflow as tf
import numpy as np
import random
import time
import copy

from collections import deque
from ComplexCar import ComplexCar
from TestingRoadv2 import Road
from GUI import GUI
from DQNetwork import DQNetwork
from DQNetwork import Memory

#GLOBAL VARIABLES + training data
length = 5
prob = 0.5
keepalive = -2
wait_weight = 0.8
numRoads = 1

state_size = 4 * (numRoads + 3)
action_size = 3
learning_rate = 0.000025

total_episodes = 2000
max_steps = 300
batch_size = 1000
num_epoch = 20

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00015

gamma = 0.95

memory_size = 200000
pretrain_length = memory_size
training = True

grid_space = 50


def create_environment():
	game = Road(numRoads = 1)

	idle = [0,1,0]
	speedUp = [0,0,1]
	slowDown = [1,0,0]
	possible_actions = [idle, speedUp, slowDown]

	return game, possible_actions


#creating environment and model parameters
game, possible_actions = create_environment()
game.newInstance()
tf.reset_default_graph()

DDQNetwork = DQNetwork(state_shape=state_size, 
					  action_shape = action_size, 
					  learning_rate=learning_rate,
					  name = "DDQNetwork")

FixedValueNetwork = DQNetwork(state_shape=state_size, 
					  action_shape = action_size, 
					  learning_rate=learning_rate,
					  name = "FixedValueNetwork")

memory = Memory(memory_size)


def update_fixed_values():

	from_DQN = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DDQNetwork")
	to_FixedValues = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "FixedValueNetwork")
	op_holder = []
	for from_DQN, to_FixedValues in zip(from_DQN, to_FixedValues):
		op_holder.append(to_FixedValues.assign(from_DQN))

	return op_holder



for i in range(pretrain_length):
	pretrain_game = pretrain[i]
	state = pretrain_game.getState()
	action = random.choice(possible_actions)
	action = np.argmax(action) - 1
	reward = pretrain_game.step(action)
	done = pretrain_game.end()
