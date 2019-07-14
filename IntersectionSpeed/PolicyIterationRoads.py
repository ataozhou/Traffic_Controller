import tensorflow as tf
import numpy as np
import random
import time
import copy
import sys

from collections import deque
from IntersectionRoads import Intersection
from GUI import GUI
from PGNetwork import PGNetwork
from DQNetwork import Memory

#GLOBAL VARIABLES + training data
numRoads = 2
length = 5
prob = 0.3
speed = 1
keepalive = -2
wait_weight = 0.7

state_size = 40
action_size = 13
learning_rate = 0.000002
#learning_rate = 0.000009

total_episodes = 2000
max_steps = 300
batch_size = 50000
epoch = 0
num_epochs = 500000

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00015

gamma = 0.95

pretrain_length = batch_size
memory_size = 5000000
grid_space = 50





def create_environment():
	game = Intersection(numRoads = numRoads, length = length, prob = prob, speed = speed, wait_weight = wait_weight)

	action1 =   [1,0,0,0]
	action2 =   [0,1,0,0]
	action3 =   [0,0,1,0]
	action4 =   [0,0,0,1]
	action5 =   [1,1,0,0]
	action5_5 = [1,1,1,0]
	action6 =   [1,0,1,0]
	action6_5 = [1,0,1,1]
	action7 =   [1,0,0,1]
	action8 =   [0,1,1,0]
	action8_5 = [0,1,1,1]
	action9 =   [0,1,0,1]
	action10 =  [0,0,1,1]

	possible_actions = [action1, action2, action3, action4, action5, action5_5, action6, action6_5, action7, action8, action8_5, action9, action10]

	return game, possible_actions

def discount_and_normalize_rewards(episode_rewards):
	discounted_episode_rewards = np.zeros_like(episode_rewards)
	cumulative = 0.0
	for i in reversed(range(len(episode_rewards))):
		cumulative = cumulative * gamma + episode_rewards[i]
		discounted_episode_rewards[i] = cumulative

	mean = np.mean(discounted_episode_rewards)
	std = np.std(discounted_episode_rewards)
	discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

	return discounted_episode_rewards

def make_batch(batch_size):
	numSteps = 0
	finished_games = 0
	states, actions, rewardsEpisode, rewardsBatch, discountedRewards = [], [], [], [], []

	episode_num = 1
	print("here")
	game.newInstance()
	state = game.getSimpleState()

	while True:
		action_probability_distribution = sess.run(PGNetwork.action_distribution, feed_dict={PGNetwork.inputs_: state.reshape(1, state_size)})
		action = np.random.choice(range(action_probability_distribution.shape[1]), p = action_probability_distribution.ravel())


		reward = game.step(possible_actions[action])
		done = game.gameEnd()

		states.append(state)
		actions.append(action)
		rewardsEpisode.append(reward)

		if done or (numSteps == 5000):
			#print(numSteps)
			sys.stdout.write("\r%d" % finished_games)
			sys.stdout.write("Game Ended at step: %d" %numSteps)
			sys.stdout.flush()
			next_state = np.zeros(state_size)
			rewardsBatch.append(rewardsEpisode)
			discountedRewards.append(discount_and_normalize_rewards(rewardsEpisode))

			if len(np.concatenate(rewardsBatch)) > batch_size:
				break
			rewardsEpisode = []
			episode_num += 1
			game.newInstance()
			state = game.getSimpleState()
			numSteps = 0
			finished_games += 1
		else:
			next_state = game.getSimpleState()
			state = next_state
			numSteps += 1
	return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewardsBatch), np.concatenate(discountedRewards), episode_num


PGNetwork = PGNetwork(state_shape = state_size, action_shape = action_size, learning_rate = learning_rate)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


game, possible_actions = create_environment()
allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
mean_reward_total = []
epoch = 1
average_reward = []

saver = tf.train.Saver()



while epoch < num_epochs + 1:
	states_mb, actions_mb, rewardsBatch, discountedRewards_mb, nbEpisodesmb = make_batch(batch_size)
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


	for i in range(len(states_mb)):
		loss_, _ = sess.run([PGNetwork.loss, PGNetwork.train_opt], feed_dict = {PGNetwork.inputs_: states_mb[i].ravel().reshape(1, state_size), 
																				PGNetwork.actions: actions_mb[i].reshape(1, action_size), 
																				PGNetwork.discounted_episode_rewards_: discountedRewards_mb[i].reshape(1)})
	print("Training Loss: {}".format(loss_))

	if epoch % 10 == 0:
		saver.save(sess, "./PGModels_1speed_2x2_13actions/model.ckpt")
		print("Model Saved")
	epoch += 1
