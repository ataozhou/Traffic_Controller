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
wait_weight = 3

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
decay_rate = 0.00015

gamma = 0.95

memory_size = 50000
pretrain_length = memory_size
grid_space = 50


def create_environment():
	game = Intersection(length=length, prob = prob, keepalive = keepalive, wait_weight=wait_weight)

	idle = [1,0,0]
	waitNS = [0,1,0]
	waitEW = [0,0,1]
	possible_actions = [idle, waitNS, waitEW]

	return game, possible_actions

def trainDQN(memory, batch_size):
	tree_idx, batch, weightsBatch = memory.sample(batch_size)
	states = np.array([each[0][0] for each in batch], ndmin = 1)
	actions = np.array([each[0][1] for each in batch])
	rewards = np.array([each[0][2] for each in batch])
	next_states = np.array([each[0][3] for each in batch])
		#print(next_states)
	done = np.array([each[0][4] for each in batch])
	target_Qs_batch = []

	target_Qs = []
	next_q_actions = []

	for each in next_states:
		temp = sess.run(DQNetwork.outputs, feed_dict={DQNetwork.inputs_: each.reshape(1,state_size)})
		target_Qs.append(temp)

	for i in range(0,len(batch)):
		terminal = done[i]

		if terminal:
				target_Qs_batch.append(rewards[i])
		else:						
			#print(target_Qs[i])
			#print(np.max(target_Qs[i]))
			target = rewards[i] + gamma * np.max(target_Qs[i])
			target_Qs_batch.append(target)

	targets = np.array([each for each in target_Qs_batch])
	_, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absErrors], feed_dict= {DQNetwork.inputs_: states,
																													 DQNetwork.target_Q: targets,
																													 DQNetwork.actions_: actions, 
																													 DQNetwork.ISWeights_: weightsBatch})

	memory.batch_update(tree_idx, absolute_errors)
	


def discount_and_normalize_rewards(reward, next_state, currentQvalue):
	nextQvalue = np.max(sess.run(DQNetwork.outputs, feed_dict = {DQNetwork.inputs_: next_state.reshape(1,state_size)}))
	discountedReward = (reward + (gamma * nextQvalue)) - currentQvalue
	return discountedReward

def make_batch(batch_size, memory):
	numSteps = 0
	finished_games = 0
	states, actions, rewardsEpisode, rewardsBatch, discountedRewards = [], [], [], [], []

	episode_num = 1
	game.newInstance()
	state = game.getState()

	while True:
		action_probability_distribution = sess.run(PGNetwork.action_distribution, feed_dict={PGNetwork.inputs_: state.reshape(1, state_size)})
		action = np.random.choice(range(action_probability_distribution.shape[1]), p = action_probability_distribution.ravel())
		action = np.array(possible_actions[action])

		#Qvalue 
		currentQvalue = sess.run(DQNetwork.Q, feed_dict = {DQNetwork.inputs_: state.reshape(1,state_size), DQNetwork.actions_: action.reshape(1, action_size)})
		reward = game.step(NS_action=action[1], EW_action=action[2])
		done = game.gameEnd()

		states.append(state)
		actions.append(action)
		rewardsEpisode.append(reward)
		

		if done or (numSteps == 5000):
			sys.stdout.write("\r%d" % finished_games)
			sys.stdout.flush()
			next_state = np.zeros(state_size)
			discountedRewards.append(discount_and_normalize_rewards(reward, next_state, currentQvalue))
			rewardsBatch.append(rewardsEpisode)
			if len(np.concatenate(rewardsBatch)) > batch_size:
				break
			rewardsEpisode = []
			episode_num += 1
			game.newInstance()
			state = game.getState()
			numSteps = 0
			finished_games += 1

		else:
			if numSteps % 1000 == 0:
				trainDQN(memory, batch_size)
			next_state = game.getState()
			experience = state.ravel(), action, reward, next_state.ravel(), done
			memory.store(experience)
			discountedRewards.append(discount_and_normalize_rewards(reward, next_state, currentQvalue))
			state = next_state
			numSteps += 1
	return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewardsBatch), np.concatenate(discountedRewards), episode_num

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

saver = tf.train.Saver()






for i in range(pretrain_length):
	if i == 0:
		state = game.getState()

	action = random.choice(possible_actions)
	reward = game.step(NS_action=action[1], EW_action=action[2])
	done = game.gameEnd()

	if done:
		next_state = game.getState()
		experience = state.ravel(), action, reward, next_state.ravel(), done 
		memory.store(experience)
		game.newInstance()
	else:
		next_state = game.getState()
		experience = state.ravel(), action, reward, next_state.ravel(), done
		memory.store(experience)
		state = next_state


while epoch < num_epochs + 1:
	states_mb, actions_mb, rewardsBatch, discountedRewards_mb, nbEpisodesmb = make_batch(batch_size, memory)
	total_reward_Batch = np.sum(rewardsBatch)
	allRewards.append(total_reward_Batch)

	meanRewardBatch = np.divide(total_reward_Batch, nbEpisodesmb)
	mean_reward_total.append(meanRewardBatch)
	average_reward_training = np.divide(np.sum(mean_reward_total), epoch)
	maximumRewardRecorded = np.amax(allRewards)

	#PRINT STATEMENTS GO HERE
	print("==========================================")
	print("Epoch: " + str(epoch) + "/" + str(num_epochs))
	print("-----------")
	print("Number of training episodes: {}".format(nbEpisodesmb))
	print("Total reward: {}".format(total_reward_Batch, nbEpisodesmb))
	print("Mean Reward of that batch {}".format(meanRewardBatch))
	print("Average Reward of all training: {}".format(average_reward_training))
	print("Max reward for a batch so far: {}".format(maximumRewardRecorded))

	'''print(len(states_mb))
	print(len(actions_mb))
	print(len(rewardsBatch))
	print(len(discountedRewards_mb))'''


	for i in range(len(states_mb)):
		loss_, _ = sess.run([PGNetwork.loss, PGNetwork.train_opt], feed_dict = {PGNetwork.inputs_: states_mb[i].ravel().reshape(1, state_size), 
																				PGNetwork.actions: actions_mb[i].reshape(1, action_size), 
																				PGNetwork.discounted_episode_rewards_: discountedRewards_mb[i].reshape(1)})
	print("Training Loss: {}".format(loss_))

	if epoch % 10 == 0:
		saver.save(sess, "./ACModelsv2/model.ckpt")
		print("Model Saved")
	epoch += 1
