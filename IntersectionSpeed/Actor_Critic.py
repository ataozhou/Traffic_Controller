import tensorflow as tf
import numpy as np
import random
import time
import copy
import sys

from collections import deque
from TestingRoadv2 import Road
from GUI import GUI
from PGNetwork import PGNetwork
from DQNetwork import DQNetwork
from DQNetwork import Memory


#GLOBAL VARIABLES + training data
length = 5
prob = 0.5
keepalive = -2
wait_weight = 3

state_size = 6
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
	game = Road(numRoads = 1)

	slowDown = [1,0,0]
	idle = [0,1,0]
	speedUp = [0,0,1]
	possible_actions = [slowDown, idle, speedUp]

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
	return loss
	


def discount_and_normalize_rewards(reward, next_state, currentQvalue):
	nextQvalue = np.max(sess.run(DQNetwork.outputs, feed_dict = {DQNetwork.inputs_: next_state.reshape(1,state_size)}))
	discountedReward = (reward + (gamma * nextQvalue)) - currentQvalue
	return discountedReward

def make_batch(batch_size, memory):
	numSteps = 0
	finished_games = 0
	states, actions, rewardsEpisode, rewardsBatch, discountedRewards = [], [], [], [], []
	DQNloss = 0
	totalLoss = 0
	episode_num = 1
	game.newInstance()
	state = game.getState()
	firstState = game.getState()

	while True:
		action_probability_distribution = sess.run(PGNetwork.action_distribution, feed_dict={PGNetwork.inputs_: state.reshape(1, state_size)})
		action = np.random.choice(range(action_probability_distribution.shape[1]), p = action_probability_distribution.ravel())
		action = np.array(possible_actions[action])
		realAction = np.argmax(action)

		#Qvalue 
		currentQvalue = sess.run(DQNetwork.Q, feed_dict = {DQNetwork.inputs_: state.reshape(1,state_size), DQNetwork.actions_: action.reshape(1, action_size)})
		reward = game.step(realAction)
		done = game.gameEnd()

		states.append(state)
		actions.append(action)
		rewardsEpisode.append(reward)
		
		DQNloss += trainDQN(memory, batch_size)
		next_state = game.getState()
		memory.store((state.ravel(), action, reward, next_state.ravel(), done))
		discountedReward = discount_and_normalize_rewards(reward, next_state, currentQvalue)
		loss_, _ = sess.run([PGNetwork.loss, PGNetwork.train_opt], feed_dict = {PGNetwork.inputs_: np.array(state).reshape(1,state_size), 
																				PGNetwork.actions: np.array(action).reshape(1, action_size), 
																				PGNetwork.discounted_episode_rewards_: np.array(discountedReward).reshape(1)})
		totalLoss += loss_
		state = next_state
		numSteps += 1

		if done or (numSteps == 5000):
			rewardsBatch.append(rewardsEpisode)
			'''if(game.getReward() < 30):
				print(firstState)
				#raw_input("Enter")
				repeatTraining(memory, firstState, episode_num, len(np.concatenate(rewardsBatch)))'''
			DQNloss += trainDQN(memory, batch_size)
			print("Episode Number: " + str(episode_num))
			print("TotalReward: " + str(game.getReward()))
			print("PGNetwork Total Loss: " + str(totalLoss))
			print("DQNetwork Total Loss: " + str(DQNloss))
			next_state = np.zeros(state_size)
			print("Progress: " + str(len(np.concatenate(rewardsBatch))))

			if len(np.concatenate(rewardsBatch)) > batch_size:
				break
			rewardsEpisode = []
			DQNloss = 0
			totalLoss = 0
			episode_num += 1
			game.newInstance()
			state = game.getState()
			firstState = game.getState()
			numSteps = 0


def repeatTraining(memory, firstState, episode_num, progress):
	game.reuseState(firstState)
	actionList = [0,0,0]
	print(game.getState())
	#raw_input("Enter")
	numTimeInRow = 0
	okfine = 0
	state = game.getState()
	DQNloss = 0
	totalLoss = 0
	while(numTimeInRow < 3 and okfine < 15):
		action_probability_distribution = sess.run(PGNetwork.action_distribution, feed_dict={PGNetwork.inputs_: state.reshape(1, state_size)})
		#print(state)
		print(action_probability_distribution)
		action = np.random.choice(range(action_probability_distribution.shape[1]), p = action_probability_distribution.ravel())
		#print(action)
		action = np.array(possible_actions[action])
		realAction = np.argmax(action) - 1
		#print(realAction)
		#print(game.getReward())
		#raw_input("Enter")

		#Qvalue 
		currentQvalue = sess.run(DQNetwork.Q, feed_dict = {DQNetwork.inputs_: state.reshape(1,state_size), DQNetwork.actions_: action.reshape(1, action_size)})
		reward = game.step(realAction)
		done = game.gameEnd()


		DQNloss += trainDQN(memory, batch_size)
		next_state = game.getState()
		memory.store((state.ravel(), action, reward, next_state.ravel(), done))
		discountedReward = discount_and_normalize_rewards(reward, next_state, currentQvalue)
		loss_, _ = sess.run([PGNetwork.loss, PGNetwork.train_opt], feed_dict = {PGNetwork.inputs_: np.array(state).reshape(1,state_size), 
																				PGNetwork.actions: np.array(action).reshape(1, action_size), 
																				PGNetwork.discounted_episode_rewards_: np.array(discountedReward).reshape(1)})
		totalLoss += loss_
		state = next_state

		if done:
			if(game.getReward() < 30):
				numTimeInRow = 0
			if(game.getReward() > 20):
				okfine += 1
				numTimeInRow +=1
			DQNloss += trainDQN(memory, batch_size)
			print("Episode Number: " + str(numTimeInRow) + " , " + str(okfine))
			print("TotalReward: " + str(game.getReward()))
			print("PGNetwork Total Loss: " + str(totalLoss))
			print("DQNetwork Total Loss: " + str(DQNloss))
			next_state = np.zeros(state_size)
			print("Progress: " + str(progress))

			DQNloss = 0
			totalLoss = 0
			game.reuseState(firstState)
			state = game.getState()
			numSteps = 0


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
	realAction = np.argmax(action) - 1
	reward = game.step(realAction)
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

	#PRINT STATEMENTS GO HERE
	print("==========================================")
	print("Epoch: " + str(epoch) + "/" + str(num_epochs))
	print("-----------")

	make_batch(batch_size, memory)
	if epoch % 10 == 0:
		saver.save(sess, "./ACModelsv2/model.ckpt")
		print("Model Saved")
	epoch += 1
