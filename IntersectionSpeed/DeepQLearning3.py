import tensorflow as tf
import numpy as np
import random
import time
import copy
import sys

from collections import deque
from Intersection import Intersection
from GUI import GUI
from DQNetwork import DQNetwork
from DQNetwork import Memory
from DQNetwork import SumTree

#GLOBAL VARIABLES + training data
length = 5
prob = 0.5
keepalive = -2
wait_weight = 2

state_size = 17
#state_size = 2*(length + 2) + 2
action_size = 3
learning_rate = 0.000025

total_episodes = 500
epoch = 0
num_epoch = 1100
max_steps = 500
batch_size = 5000
updateModelMax = 10000

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0005

gamma = 0.95

memory_size = 200000
pretrain_length = memory_size
training = True

grid_space = 50


def create_environment():
	game = Intersection(length=length, prob = prob, keepalive = keepalive, wait_weight=wait_weight)

	idle = [1,0,0]
	waitNS = [0,1,0]
	waitEW = [0, 0,1]
	possible_actions = [idle, waitNS, waitEW]

	return game, possible_actions




#creating environment and model parameters
game, possible_actions = create_environment()
tf.reset_default_graph()

DDQNetwork = DQNetwork(state_shape = state_size, 
					  action_shape = action_size, 
					  learning_rate = learning_rate, 
					  name = "DDQNetwork")

FixedValueNetwork = DQNetwork(state_shape = state_size, 
					  action_shape = action_size, 
					  learning_rate = learning_rate, 
					  name = "FixedValueNetwork")


def update_fixed_values():

	from_DQN = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DDQNetwork")

	to_FixedValues = tf. get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "FixedValueNetwork")

	op_holder = []

	for from_DQN, to_FixedValues in zip(from_DQN, to_FixedValues):
		op_holder.append(to_FixedValues.assign(from_DQN))

	return op_holder


game.newInstance()
memory = Memory(capacity = memory_size)


#PRE-TRAINING
for i in range(pretrain_length):
	if i == 0:
		state = game.getBinary()

	action = random.choice(possible_actions)
	reward = game.step(NS_action=action[0], EW_action=action[1])
	done = game.gameEnd()

	if done:
		sys.stdout.write("\r%d" % i)
		sys.stdout.flush()
		next_state = game.getBinary()
		experience = state.ravel(), action, reward, next_state.ravel(), done
		memory.store(experience)
		game.newInstance()
		state = game.getBinary()
	else:
		next_state = game.getBinary()
		experience = state.ravel(), action, reward, next_state.ravel(), done
		memory.store(experience)
		state = next_state

#writer = tf.summary.FileWriter("/tensorboard/dqn/1")
tf.summary.scalar("Loss", DDQNetwork.loss)

write_op = tf.summary.merge_all()


saver = tf.train.Saver()

if training == True:
	rewards_list = []

#Training the Neural Net
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	game.newInstance()

	decay_step = 0
	explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
	updateModel = 0
	totalReward_Episode = []


	update_target = update_fixed_values()
	sess.run(update_target)

	while epoch < num_epoch:
		explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
		decay_step += 5

		for episode in range(total_episodes):
			game.newInstance()
			step = 0

			state = game.getBinary()
			rewards_episode = []
			
			#window = GUI(grid_space)

			while step < max_steps:
				step +=1
				updateModel += 1

				exp_exp_tradeoff = np.random.rand()

				if(explore_probability > exp_exp_tradeoff):
					action = random.choice(possible_actions)

				else:
					Qs = sess.run(DDQNetwork.outputs, feed_dict = {DDQNetwork.inputs_: state.reshape(1,state_size)})
					action = np.argmax(Qs)

					action = possible_actions[int(action)]

				reward = game.step(NS_action=action[0], EW_action=action[1])
				#window.update(game.getState())
				done = game.gameEnd()

				if done:
					sys.stdout.write("\r%d" % episode)
					sys.stdout.flush()
					next_state = game.getBinary()
					step = max_steps
					total_reward = game.getReward()
					totalReward_Episode.append(total_reward)
					experience = state.ravel(), action, reward, next_state.ravel(), done
					memory.store(experience)
					
				else:
					next_state = game.getBinary()
					experience = state.ravel(), action, reward, next_state.ravel(), done
					memory.store(experience)
					state = next_state


			#Learning
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
		_, loss, absolute_errors = sess.run([DDQNetwork.optimizer, DDQNetwork.loss, DDQNetwork.absErrors], feed_dict= {DDQNetwork.inputs_: states,
																													 DDQNetwork.target_Q: targets,
																													 DDQNetwork.actions_: actions, 
																													 DDQNetwork.ISWeights_: weightsBatch})

		memory.batch_update(tree_idx, absolute_errors)

		summary = sess.run(write_op, feed_dict = {DDQNetwork.inputs_: states,
												DDQNetwork.target_Q: targets, 
												DDQNetwork.actions_: actions, 
												DDQNetwork.ISWeights_: weightsBatch})
					#writer.add_summary(summary,episode)
					#writer.flush()

		print('Epoch: {}'.format(epoch),
			  'Average Reward: {}'.format(np.mean(totalReward_Episode)),
			  'Training loss: {:.4f}'.format(loss),
			  'Explore P: {:.4f}'.format(explore_probability))
		if epoch % 100 == 0:
			updateModel = 0
			update_target = update_fixed_values()
			sess.run(update_target)
			print("Model Updated")
		
		#saving models
		if epoch % 100 == 0:
			save_path = saver.save(sess, "./DQmodels/model.ckpt")
			print("Model Saved")

		epoch +=1
			
					



