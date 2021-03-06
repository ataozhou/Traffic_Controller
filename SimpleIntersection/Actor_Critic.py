import tensorflow as tf
import numpy as np
import random
import time
import copy
import sys

from collections import deque
from Intersection import Intersection
from GUI import GUI
from PGNetwork import PGNetwork
from DQNetwork import DQNetwork
from DQNetwork import Memory


#GLOBAL VARIABLES + training data
length = 5
prob = 0.5
keepalive = -2
wait_weight = 0.8

state_size = 4*(length + 2)
action_size = 3
learning_rate = 0.000002

total_episodes = 2000
max_steps = 300
batch_size = 5000
epoch = 0
num_epochs = 1000

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0005

gamma = 0.95

pretrain_length = batch_size
memory_size = 5000000000
grid_space = 50


def create_environment():
	game = Intersection(length=length, prob = prob, keepalive = keepalive, wait_weight=wait_weight)

	idle = [1,0,0]
	waitNS = [0,1,0]
	waitEW = [0,0,1]
	possible_actions = [idle, waitNS, waitEW]

	return game, possible_actions

def trainDQN(memory, batch_size):
	batches = memory.minisample(batch_size)
	states, actions, rewards, next_states, dones = [], [], [], [], []
	for batch in batches:
		states.append(batch[0])
		actions.append(batch[1])
		rewards.append(batch[2])
		next_states.append(batch[3])
		dones.append(batch[4])
	targets_Qs_batch = []
	target_Qs = []
	for each in next_states:
		temp = sess.run(DQNetwork.outputs, feed_dict = {DQNetwork.inputs_: each.reshape(1,state_size)})
		target_Qs.append(temp)
	for i in range(0, len(batches)):
		terminal = dones[i]
		if terminal:
			targets_Qs_batch.append(rewards[i])
		else:
			target = rewards[i] + gamma * np.max(target_Qs[i])
			targets_Qs_batch.append(target)

	targets = np.array([each for each in targets_Qs_batch])
	totalLoss = 0
	for i in range(batch_size):
		loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
						feed_dict= {DQNetwork.inputs_: states[i].reshape(1,state_size),
									DQNetwork.target_Q: targets[i],
									DQNetwork.actions_: np.array(actions[i]).reshape(1, action_size)})
		totalLoss += loss
	return totalLoss


def discount_and_normalize_rewards(action, reward, next_state, currentQvalue):
	nextQvalue = sess.run(DQNetwork.Q, feed_dict = {DQNetwork.inputs_: state.reshape(1,state_size), DQNetwork.actions_: action.reshape(1, action_size)})
	discountedReward = (reward + (gamma * nextQvalue)) - currentQvalue
	return discountedReward

def make_batch(batch_size, memory):
	numSteps = 0
	#states, actions, 
	rewardsEpisode, rewardsBatch = [], []

	episode_num = 1
	DQNloss = 0
	totalLoss = 0
	game.newInstance()
	state = game.getState()

	while True:
		action_probability_distribution = sess.run(PGNetwork.action_distribution, feed_dict={PGNetwork.inputs_: state.reshape(1, state_size)})
		action = np.random.choice(range(action_probability_distribution.shape[1]), p = action_probability_distribution.ravel())
		action = np.array(possible_actions[action])

		#Qvalue of state before action
		currentQvalue = sess.run(DQNetwork.Q, feed_dict = {DQNetwork.inputs_: state.reshape(1,state_size), DQNetwork.actions_: action.reshape(1, action_size)})
		reward = game.step(NS_action=action[1], EW_action=action[2])
		done = game.gameEnd()

		'''states.append(state)
		actions.append(action)'''
		rewardsEpisode.append(reward)

		if done or (numSteps == 5000):
			DQNloss += trainDQN(memory, batch_size)
			print("Episode Number: " + str(episode_num))
			print("TotalReward: " + str(game.getReward()))
			print("PGNetwork Total Loss: " + str(totalLoss))
			print("DQNetwork Total Loss: " + str(DQNloss))
			next_state = np.zeros(state_size)
			rewardsBatch.append(rewardsEpisode)
			print("Progress: " + str(len(np.concatenate(rewardsBatch))))

			if len(np.concatenate(rewardsBatch)) > batch_size:
				break
			rewardsEpisode = []
			DQNloss = 0
			totalLoss = 0
			episode_num += 1
			game.newInstance()
			state = game.getState()
			numSteps = 0

		if numSteps % 1000 == 0:
			DQNloss += trainDQN(memory, batch_size)
		next_state = game.getState()
		memory.add((state.ravel(), action, reward, next_state.ravel(), done))
		discountedReward = discount_and_normalize_rewards(action, reward, next_state, currentQvalue)
		loss_, _ = sess.run([PGNetwork.loss, PGNetwork.train_opt], feed_dict = {PGNetwork.inputs_: np.array(state).reshape(1,state_size), 
																				PGNetwork.actions: np.array(action).reshape(1, action_size), 
																				PGNetwork.discounted_episode_rewards_: np.array(discountedReward).reshape(1)})
		totalLoss += loss_
		state = next_state
		numSteps += 1

PGNetwork = PGNetwork(state_shape = state_size, 
					  action_shape = action_size, 
					  learning_rate = learning_rate)


DQNetwork = DQNetwork(state_shape=state_size, 
					  action_shape = action_size, 
					  learning_rate=learning_rate,
					  name = "DQNetwork")

memory = Memory(memory_size)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


game, possible_actions = create_environment()
allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
mean_reward_total = []
average_reward = []




game.newInstance()
memory = Memory(max_size = memory_size)

saver = tf.train.Saver()






for i in range(pretrain_length):
	if i == 0:
		state = game.getState()

	action = random.choice(possible_actions)
	reward = game.step(NS_action=action[1], EW_action=action[2])
	done = game.gameEnd()

	if done:
		next_state = game.getState()
		memory.add((state.ravel(), action, reward, next_state.ravel(), done))
		game.newInstance()
	else:
		next_state = game.getState()
		memory.add((state.ravel(), action, reward, next_state.ravel(), done))
		state = next_state


while epoch < num_epochs:
	print("EPOCH " + str(epoch) + "=========================")
	make_batch(batch_size, memory)
	
	if epoch % 10 == 0:
		saver.save(sess, "./ACmodels_EP/model.ckpt")
		print("Model Saved")
	epoch +=1




