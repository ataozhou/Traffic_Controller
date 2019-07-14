import tensorflow as tf
import numpy as np
import random
import time
import copy
import sys

from collections import deque
from IntersectionDirection import Intersection
from GUI import GUI
from PGNetwork import PGNetwork
from DQNetwork import Memory

#GLOBAL VARIABLES + training data
numRoads = 1
length = [3, 48, 498]
prob = 0.1
speed = 1
keepalive = -2
wait_weight = 0.7

state_size = 20
action_size = 3
learning_rate = 0.000002
numTrials = 100
#learning_rate = 0.000009

max_steps = 4500

grid_space = 50


def create_environment():
	
	games =  [Intersection(numRoads = numRoads, length = length[0], prob = prob, speed = speed, keepalive = keepalive, wait_weight = wait_weight) for i in range(4)]
	games += [Intersection(numRoads = numRoads, length = length[1], prob = prob, speed = speed, keepalive = keepalive, wait_weight = wait_weight) for i in range(4)]
	games += [Intersection(numRoads = numRoads, length = length[2], prob = prob, speed = speed, keepalive = keepalive, wait_weight = wait_weight) for i in range(4)]

	for i in range(len(games)):
		print(id(games[i]))

	idle = [1,0,0]
	NSGo = [0,1,0]
	EWGo = [0,0,1]
	possible_actions = [idle, NSGo, EWGo]

	return games, possible_actions



with tf.Session() as sess:

	PGNetwork = PGNetwork( 
						  state_shape=state_size, 
						  action_shape = action_size, 
						  learning_rate=learning_rate,
						  name = "PGNetwork")

	games, possible_actions = create_environment()

	avgWait = [0] * 12
	trafficPerc = [0] * 12
	avgReward = [0] * 12
	totalReward = [0] * 12

	probAvgWait = [0] * 12
	probAvgTrafficPerc = [0] * 12

	saver = tf.train.Saver()

	saver.restore(sess, "./PGModels_1speed_1x1_0.3/model.ckpt")
	#window = GUI(grid_space)

	step = 0
	for j in range(len(games)):

		games[j].newInstance()
		avgWait[j] = [0] * max_steps
		trafficPerc[j] = [0] * max_steps
		avgReward[j] = [0] * max_steps

		probAvgWait[j] = [0] * 9
		probAvgTrafficPerc[j] = [0] * 9

		if (j % 4 == 0):
			for i in range(numTrials):
				games[j].newInstance()
				saver.restore(sess, "./PGModels_1speed_1x1_0.3/model.ckpt")
				sys.stdout.write("\r%d" % (i + 1))
				sys.stdout.flush()
				step = 0
				totalScore = 0
				while (not games[j].gameEnd()) and (step < max_steps):
					if games[j].NSRoads[0].prob == 0.5:
						saver.restore(sess, "./PGModels_1speed_1x1(3)_0.5/model.ckpt")
					frame = games[j].getSimpleState()
					action_probability_distribution = sess.run(PGNetwork.action_distribution, feed_dict={PGNetwork.inputs_: frame.reshape(1, state_size)})
					action = np.argmax(action_probability_distribution)
					action = possible_actions[action]
					reward = games[j].step(action[1], action[2])
					avgReward[j][step] += (reward/numTrials)
					avgWait[j][step] += (games[j].getWait()/numTrials)
					trafficPerc[j][step] += (games[j].trafficSaturation())/ numTrials
					#window.update(j.binaryRepresentation())
					#print("Score: ", reward)
					totalScore += reward/numTrials
					step += 1
				totalReward[j] += totalScore/numTrials
			print("GAME END, Index: " + str(j) + " TOTAL_SCORE:", totalScore)


		if (j % 4 == 1):
			for i in range(numTrials):
				games[j].newInstance()
				sys.stdout.write("\r%d" % (i + 1))
				sys.stdout.flush()
				step = 0
				totalScore = 0
				while (not games[j].gameEnd()) and (step < max_steps):
					reward = games[j].step(1,0)
					avgReward[j][step] += (reward/numTrials)
					avgWait[j][step] += (games[j].getWait()/numTrials)
					trafficPerc[j][step] += (games[j].trafficSaturation())/ numTrials
					#window.update(j.binaryRepresentation())
					#print("Score: ", reward)
					totalScore += reward/numTrials
					step += 1
				totalReward[j] += totalScore/numTrials
			print("GAME END, Index: " + str(j) + " TOTAL_SCORE:", totalScore)


		elif (j % 4 == 2):
			for i in range(numTrials):
				games[j].newInstance()
				sys.stdout.write("\r%d" % (i + 1))
				sys.stdout.flush()
				step = 0
				totalScore = 0
				while (not games[j].gameEnd()) and (step < max_steps):
					
					action = games[j].chooseRoad2()
					reward = games[j].step(action[0], action[1])
					avgReward[j][step] += (reward/numTrials)
					avgWait[j][step] += (games[j].getWait()/numTrials)
					trafficPerc[j][step] += (games[j].trafficSaturation())/ numTrials
					#window.update(j.binaryRepresentation())
					#print("Score: ", reward)
					totalScore += reward/numTrials
					step += 1
				totalReward[j] += totalScore/numTrials
			print("GAME END, Index: " + str(j) + " TOTAL_SCORE:", totalScore)


		elif (j % 4 == 3):
			for i in range(numTrials):
				games[j].newInstance()
				sys.stdout.write("\r%d" % (i + 1))
				sys.stdout.flush()
				step = 0
				totalScore = 0
				while (not games[j].gameEnd()) and (step < max_steps):
					
					action = games[j].chooseRoad3()
					reward = games[j].step(action[0], action[1])
					avgReward[j][step] += (reward/numTrials)
					avgWait[j][step] += (games[j].getWait()/numTrials)
					trafficPerc[j][step] += (games[j].trafficSaturation())/ numTrials
					#window.update(j.binaryRepresentation())
					#print("Score: ", reward)
					totalScore += reward/numTrials
					step += 1
				totalReward[j] += totalScore/numTrials
			print("GAME END, Index: " + str(j) + " TOTAL_SCORE:", totalScore)


	index = 0
	for i in range(len(games)):
		start = 0
		index = 0
		waitVal = 0
		trafficVal = 0
		while start != max_steps:
			waitVal = 0
			trafficVal = 0
			for j in range(start, (start + 500)):
				print(j)
				waitVal += float(avgWait[i][j]/ 500)
				trafficVal += float(trafficPerc[i][j]/ 500)
			print("\n\n\n\n")
			probAvgWait[i][index] = float(waitVal)
			probAvgTrafficPerc[i][index] = float(trafficVal)
			start += 500
			index += 1



	f = open('Stats2_3', 'w')
	f.write("Average Total Reward\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	f.write(str(totalReward[0]) + "\t" + str(totalReward[1]) + "\t" + str(totalReward[2])+ "\n" + str(totalReward[3]) + "\n\n")


	f.write("Average Wait Per Probability\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	for i in range(index):
		f.write(str(probAvgWait[0][i]) + "\t" + str(probAvgWait[1][i]) + "\t" + str(probAvgWait[2][i])+"\t" + str(probAvgWait[3][i])+"\n")
	f.write("\n")

	f.write("Average Traffic Percent Per Probability\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	for i in range(index):
		f.write(str(probAvgTrafficPerc[0][i]) + "\t" + str(probAvgTrafficPerc[1][i]) + "\t" + str(probAvgTrafficPerc[2][i])+"\t" + str(probAvgTrafficPerc[3][i])+"\n")
	f.write("\n")


	'''f.write("Average Reward\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	for i in range(max_steps):
		f.write(str(avgReward[0][i]) + "\t" + str(avgReward[1][i]) + "\t" + str(avgReward[2][i])+ "\t" + str(avgReward[3][i]) +"\n")
	f.write("\n")'''

	f.write("Average Wait\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	for i in range(max_steps):
		f.write(str(avgWait[0][i]) + "\t" + str(avgWait[1][i]) + "\t" + str(avgWait[2][i])+"\t" + str(avgWait[3][i])+"\n")
	f.write("\n")

	f.write("Average Traffic Percent\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	for i in range(max_steps):
		f.write(str(trafficPerc[0][i]) + "\t" + str(trafficPerc[1][i]) + "\t" + str(trafficPerc[2][i])+"\t" + str(trafficPerc[3][i])+"\n")
	f.write("\n")
	f.close()



	f = open('Stats2_48', 'w')
	f.write("Average Total Reward\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	f.write(str(totalReward[4]) + "\t" + str(totalReward[5]) + "\t" + str(totalReward[6])+"\t" + str(totalReward[7])+"\n\n")


	f.write("Average Wait Per Probability\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	for i in range(index):
		f.write(str(probAvgWait[4][i]) + "\t" + str(probAvgWait[5][i]) + "\t" + str(probAvgWait[6][i])+"\t" + str(probAvgWait[7][i])+"\n")
	f.write("\n")

	f.write("Average Traffic Percent Per Probability\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	for i in range(index):
		f.write(str(probAvgTrafficPerc[4][i]) + "\t" + str(probAvgTrafficPerc[5][i]) + "\t" + str(probAvgTrafficPerc[6][i])+"\t" + str(probAvgTrafficPerc[7][i])+"\n")
	f.write("\n")

	'''f.write("Average Reward\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	for i in range(max_steps):
		f.write(str(avgReward[4][i]) + "\t" + str(avgReward[5][i]) + "\t" + str(avgReward[6][i])+"\t" + str(avgReward[7][i])+"\n")
	f.write("\n")'''

	f.write("Average Wait\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	for i in range(max_steps):
		f.write(str(avgWait[4][i]) + "\t" + str(avgWait[5][i]) + "\t" + str(avgWait[6][i])+"\t" + str(avgWait[7][i])+"\n")
	f.write("\n")

	f.write("Average Traffic Percent\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	for i in range(max_steps):
		f.write(str(trafficPerc[4][i]) + "\t" + str(trafficPerc[5][i]) + "\t" + str(trafficPerc[6][i])+"\t" + str(trafficPerc[7][i])+"\n")
	f.write("\n")
	f.close()



	f = open('Stats2_498', 'w')
	f.write("Average Total Reward\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	f.write(str(totalReward[8]) + "\t" + str(totalReward[9]) + "\t" + str(totalReward[10])+"\t" + str(totalReward[11])+"\n\n")

	f.write("Average Wait Per Probability\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	for i in range(index):
		f.write(str(probAvgWait[8][i]) + "\t" + str(probAvgWait[9][i]) + "\t" + str(probAvgWait[10][i])+"\t" + str(probAvgWait[11][i])+"\n")
	f.write("\n")

	f.write("Average Traffic Percent Per Probability\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	for i in range(index):
		f.write(str(probAvgTrafficPerc[8][i]) + "\t" + str(probAvgTrafficPerc[9][i]) + "\t" + str(probAvgTrafficPerc[10][i])+"\t" + str(probAvgTrafficPerc[11][i])+"\n")
	f.write("\n")

	'''f.write("Average Reward\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	for i in range(max_steps):
		f.write(str(avgReward[8][i]) + "\t" + str(avgReward[9][i]) + "\t" + str(avgReward[10][i])+"\t" + str(avgReward[11][i])+"\n")
	f.write("\n")'''

	f.write("Average Wait\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	for i in range(max_steps):
		f.write(str(avgWait[8][i]) + "\t" + str(avgWait[9][i]) + "\t" + str(avgWait[10][i])+"\t" + str(avgWait[11][i])+"\n")
	f.write("\n")

	f.write("Average Traffic Percent\n")
	f.write("NeuralNet\tNS\t50/50\tMOSTCARS\n")
	for i in range(max_steps):
		f.write(str(trafficPerc[8][i]) + "\t" + str(trafficPerc[9][i]) + "\t" + str(trafficPerc[10][i])+"\t" + str(trafficPerc[11][i])+"\n")
	f.write("\n")
	f.close()
