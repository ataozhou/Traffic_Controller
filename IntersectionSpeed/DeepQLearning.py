import tensorflow as tf
import numpy as np
import random
import time
import copy
import sys

from collections import deque
from ComplexCar import ComplexCar
from TestingRoad import Road
from GUI import GUI
from DQNetwork import DQNetwork
from DQNetwork import Memory

#GLOBAL VARIABLES + training data
length = 5
prob = 0.5
keepalive = -2
wait_weight = 0.8
numRoads = 1

state_size = 6 * (numRoads + 3)
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

starting_states = []

for i in range(0,3):
	for j in range(0, 3):
		copy = Road(numRoads = 1)
		copy.newInstance()
		copy.putCar(1, 2, ComplexCar(0, i))
		copy.putCar(0, 2, ComplexCar(0, j))
		starting_states.append(copy)

for i in range(0,3):
	for j in range(0, 3):
		copy = Road(numRoads = 1)
		copy.newInstance()
		copy.putCar(1, 3, ComplexCar(0, i))
		copy.putCar(0, 2, ComplexCar(0, j))
		starting_states.append(copy)

for i in range(0,3):
	for j in range(0, 3):
		copy = Road(numRoads = 1)
		copy.newInstance()
		copy.putCar(1, 2, ComplexCar(0, i))
		copy.putCar(0, 3, ComplexCar(0, j))
		starting_states.append(copy)

for i in range(0,3):
	for j in range(0, 3):
		copy = Road(numRoads = 1)
		copy.newInstance()
		copy.putCar(1, 3, ComplexCar(0, i))
		copy.putCar(0, 3, ComplexCar(0, j))
		starting_states.append(copy)

for i in range(0,3):
	for j in range(0, 3):
		copy = Road(numRoads = 1)
		copy.newInstance()
		copy.putCar(1, 2, ComplexCar(0, i))
		copy.putCar(0, 2, ComplexCar(0, j))
		copy.putCar(0, 3, ComplexCar(0, j))
		starting_states.append(copy)

for i in range(0,3):
	for j in range(1, 3):
		copy = Road(numRoads = 1)
		copy.newInstance()
		copy.putCar(1, 2, ComplexCar(0, i))
		copy.putCar(0, 2, ComplexCar(0, j))
		copy.putCar(0, 3, ComplexCar(0, j - 1))
		starting_states.append(copy)

for i in range(0,3):
	for j in range(0, 3):
		copy = Road(numRoads = 1)
		copy.newInstance()
		copy.putCar(1, 3, ComplexCar(0, i))
		copy.putCar(0, 2, ComplexCar(0, j))
		copy.putCar(0, 3, ComplexCar(0, j))
		starting_states.append(copy)

for i in range(0,3):
	for j in range(1, 3):
		copy = Road(numRoads = 1)
		copy.newInstance()
		copy.putCar(1, 3, ComplexCar(0, i))
		copy.putCar(0, 2, ComplexCar(0, j))
		copy.putCar(0, 3, ComplexCar(0, j - 1))
		starting_states.append(copy)

size = len(starting_states)
index = np.random.choice(np.arange(size), size = pretrain_length, replace = True)
pretrain = [starting_states[i] for i in index]
currentTest = 0
pretrain_game = pretrain[currentTest]

for i in range(pretrain_length):

	state = pretrain_game.getState()
	action = random.choice(possible_actions)
	action = np.argmax(action) - 1
	reward = pretrain_game.step(action)
	done = pretrain_game.end()

	if done:
		sys.stdout.write("\r%d" % i)
		sys.stdout.flush()
		next_state = game.getState()
		experience = state.ravel(), action, reward, next_state.ravel(), done
		memory.store(experience)
		currentTest += 1
		pretrain_game = pretrain[currentTest]
		state = pretrain_game.getState()
	else:
		next_state = game.getBinary()
		experience = state.ravel(), action, reward, next_state.ravel(), done
		memory.store(experience)
		state = next_state