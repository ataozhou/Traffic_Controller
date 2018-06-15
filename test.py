import tensorflow as tf
import numpy as np
import random
import time
import copy

from collections import deque
from Intersection import Intersection
from GUI import GUI
from NeuralNet import NeuralNet
from NeuralNet import Memory

#GLOBAL VARIABLES + training data
length = 5
prob = 0.5
keepalive = -0.5
wait_weight = 0.5

state_size = 2*(length + 2)
action_size = 3
learning_rate = 0.0002

total_episodes = 5000
max_steps = 300
batch_size = 1000
num_epoch = 20

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

gamma = 0.99

pretrain_length = batch_size
memory_size = 5000000
training = True

grid_space = 50


def create_environment():
	game = Intersection(length=length, prob = prob, keepalive = keepalive, wait_weight=wait_weight)

	idle = [1,0,0]
	waitNS = [0,1,0]
	waitEW = [0,0,1]
	possible_actions = [idle, waitNS, waitEW]

	return game, possible_actions


#creating environment and model parameters
game, possible_actions = create_environment()
tf.reset_default_graph()

DQNetwork = NeuralNet(batch_size=batch_size, 
					  state_shape=state_size, 
					  action_shape = action_size, 
					  learning_rate=learning_rate)


game.newInstance()
memory = Memory(max_size = memory_size)


#PRE-TRAINING
for i in range(pretrain_length):
	if i == 0:
		state = game.getState()

	action = random.choice(possible_actions)
	reward = game.step(NS_action=action[1], EW_action=action[2])
	done = game.gameEnd()

	if done:
		print("Game Ended at step" + str(i))
		next_state = game.getState()
		memory.add((state.ravel(), action, reward, next_state.ravel(), done))
		game.newInstance()
	else:
		next_state = game.getState()
		memory.add((state.ravel(), action, reward, next_state.ravel(), done))
		state = next_state

writer = tf.summary.FileWriter("/tensorboard/dqn/1")
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()


saver = tf.train.Saver()

if training == True:
	rewards_list = []

#Training the Neural Net
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	game.newInstance()

	decay_step = 0

	for episode in range(total_episodes):
		game.newInstance()
		step = 0

		state = game.getState()

		window = GUI(grid_space)

		while step < max_steps:
			step +=1
			decay_step += 1

			exp_exp_tradeoff = np.random.rand()

			explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

			if(explore_probability > exp_exp_tradeoff):
				action = random.choice(possible_actions)

			else:
				Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.ravel()})

				action = np.argmax(Qs)

				action = possible_actions[int(action)]

			reward = game.step(NS_action=action[1], EW_action=action[2])
			window.update(game.getState())

			done = game.gameEnd()

			if done:
				next_state = game.getState()
				step = max_steps
				total_reward = game.getReward()
				print('Episode: {}'.format(episode),
					'Total reward: {}'.format(total_reward),
					'Training loss: {:.4f}'.format(loss),
					'Explore P: {:.4f}'.format(explore_probability))

				rewards_list.append((episode, total_reward))
				memory.add((state.ravel(), action, reward, next_state.ravel(), done))

			else:
				next_state = game.getState()
				memory.add((state.ravel(), action, reward, next_state.ravel(), done))
				state = next_state


			#Learning
			batch = memory.sample(batch_size)
			states = np.array([each[0] for each in batch], ndmin = 1)
			actions = np.array([each[1] for each in batch])
			rewards = np.array([each[2] for each in batch])
			next_states = np.array([each[3] for each in batch])
			done = np.array([each[4] for each in batch])

			target_Qs_batch = []

			target_Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states})


			for i in range(0,len(batch)):
				terminal = done[i]

				if terminal:
					target_Qs_batch.append(rewards[i])
				else:
					target = rewards[i] + gamma * np.max(target_Qs[i])
					target_Qs_batch.append(target)

			targets = np.array([each for each in target_Qs_batch])

			loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
				feed_dict= {DQNetwork.inputs_: states,
							DQNetwork.target_Q: targets,
							DQNetwork.actions_: actions})
			summary = sess.run(write_op, feed_dict = {DQNetwork.inputs_: states,
													DQNetwork.target_Q: targets, 
													DQNetwork.actions_: actions})
			writer.add_summary(summary,episode)
			writer.flush()

			#saving models
			if episode % 5 == 0:
				save_path = saver.save(sess, "./models/model.ckpt")
				print("Model Saved")



