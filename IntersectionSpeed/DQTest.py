import tensorflow as tf
import numpy as np
import random
import time
import copy
import sys

from collections import deque
from Car import Car
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

state_size = 18
action_size = 3
learning_rate = 0.000000025

total_episodes = 500
max_steps = 300
batch_size = 1000
num_epoch = 20

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0015

gamma = 0.95

memory_size = 100000
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


def repeatTraining(memory, firstState, episode_num, progress):
	numTrain = 0
	okfine = 0

	while(numTrain < 3 and okfine < 50):
		chance = random.random()
		if(chance < 0.333):
			action = [1,0,0]
		elif(chance < 0.6666):
			action = [0,1,0]
		elif(chance < 1):
			action = [0,0,1]


for i in range(pretrain_length):

	state = game.getState()
	action = random.choice(possible_actions)
	realAction = np.argmax(action) - 1
	reward = game.step(realAction)
	done = game.gameEnd()

	if done:
		next_state = game.getState()
		experience = state.ravel(), action, reward, next_state.ravel(), done
		memory.store(experience)
		state = game.getState()
	else:
		next_state = game.getState()
		experience = state.ravel(), action, reward, next_state.ravel(), done
		memory.store(experience)
		state = next_state
	sys.stdout.write("\r%d" % i)
	sys.stdout.flush()



#writer = tf.summary.FileWriter("/tensorboard/DQN/1")
tf.summary.scalar("Loss", DDQNetwork.loss)
write_op = tf.summary.merge_all()
saver = tf.train.Saver()

if training == True:
	rewards_list = []



#Training the Neural Net
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	decay_step = 0
	updateModel = 0
	totalReward_Episode = []
	game.newInstance()
	#print(game.getState())
	update_target = update_fixed_values()
	sess.run(update_target)

	for episode in range(total_episodes):
		game.newInstance()
		#print(game.getReward())
		step = 0
		state = game.getState()
		firstState = game.getState()
		rewards_episode = []
			
			#window = GUI(grid_space)

		while step < max_steps:
			#sys.stdout.write("\r%d" % step)
			#sys.stdout.flush()
			#print(game.getState())
			step +=1
			updateModel += 1
			decay_step += 1
			exp_exp_tradeoff = np.random.rand()
			explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
			if(explore_probability > exp_exp_tradeoff):
				action = random.choice(possible_actions)


			else:
				Qs = sess.run(DDQNetwork.outputs, feed_dict = {DDQNetwork.inputs_: state.reshape(1,state_size)})
				action = np.argmax(Qs)
				action = possible_actions[int(action)]

			realAction = np.argmax(action) - 1
			reward = game.step(realAction)
			#print("Step Reward: " + str(reward))
			#window.update(game.getState())
			dones = game.gameEnd()


			if dones:
				next_state = game.getState()
				step = max_steps
				total_reward = game.getReward()
				print("Total Reward: " + str(total_reward))
				'''if(total_reward < 30):
					repeatTraining(memory, firstState, episode)'''
				totalReward_Episode.append(total_reward)
				experience = state.ravel(), action, reward, next_state.ravel(), dones
				memory.store(experience)
				game.newInstance()
				firstState = game.getState()
					
			else:
				next_state = game.getState()
				experience = state.ravel(), action, reward, next_state.ravel(), dones
				memory.store(experience)
				state = next_state


			tree_idx, batch, weightsBatch = memory.sample(batch_size)
			states = np.array([each[0][0] for each in batch], ndmin = 1)
			actions = np.array([each[0][1] for each in batch])
			rewards = np.array([each[0][2] for each in batch])
			next_states = np.array([each[0][3] for each in batch])
			done = np.array([each[0][4] for each in batch])
			target_Qs_batch = []

			target_Qs = []
			next_q_actions = []

			for each in next_states:
				next_action = np.argmax(sess.run(DDQNetwork.outputs, feed_dict={DDQNetwork.inputs_: each.reshape(1,state_size)}))
				temp = sess.run(FixedValueNetwork.outputs, feed_dict={FixedValueNetwork.inputs_: each.reshape(1,state_size)})
				temp = temp.ravel()
				target_Qs.append(temp)
				next_q_actions.append(next_action)

			for i in range(0,len(batch)):
				terminal = done[i]

				if terminal:
					target_Qs_batch.append(rewards[i])
				else:						
				#print(target_Qs[i])
				#print(np.max(target_Qs[i]))
					action = next_q_actions[i]
					target = rewards[i] + gamma * target_Qs[i][action]
					target_Qs_batch.append(target)

			targets = np.array([each for each in target_Qs_batch])
			#print(weightsBatch)
			for i in range(len(batch)):
				_, loss, absolute_errors = sess.run([DDQNetwork.optimizer, DDQNetwork.loss, DDQNetwork.absErrors], feed_dict= {DDQNetwork.inputs_: states[i].reshape(1, state_size), 
																															DDQNetwork.target_Q: targets[i], 
																															DDQNetwork.actions_: np.array(actions[i]).reshape(1, action_size),  
																															DDQNetwork.ISWeights_: weightsBatch[i].reshape(1,1)})

			memory.batch_update(tree_idx, absolute_errors)

			if dones:
				print('Episode: {}'.format(episode), 'Total reward: {}'.format(total_reward), 'Training loss: {:.4f}'.format(loss), 'Explore P: {:.4f}'.format(explore_probability))


			if updateModel > 50:
				update_target = update_fixed_values()
				sess.run(update_target)
				updateModel = 0
				print("Model Updated")


		if( episode % 10 == 0):
			save_path = saver.save(sess, "./model_1r/model.ckpt")
			print("Model Saved")