
from Car import Car
from Road import Road
import numpy as np



class Intersection:

	def __init__(self, length, prob, keepalive, wait_weight):
		self.NSRoad = Road(length,prob)
		self.EWRoad = Road(length,prob)
		self.keepalive = keepalive
		self.total_reward = 0
		self.length = length
		self.prob = prob
		self.end = False
		self.wait_weight = wait_weight

	def getState(self):
		temp = np.array([np.array(self.NSRoad.binaryRepresentation()), np.array(self.EWRoad.binaryRepresentation())])
		return temp

	def newInstance(self):
		self.NSRoad = Road(self.length, self.prob)
		self.EWRoad = Road(self.length, self.prob)
		self.end = False
		self.total_reward = 0

	def getReward(self):
		return self.total_reward

	def gameEnd(self):
		return self.end

	def step(self,NS_action, EW_action):
		NSoutput = self.NSRoad.step(NS_action)
		EWoutput = self.EWRoad.step(EW_action)
		step_reward = 0

		if (NSoutput[1][0] == 1) and (EWoutput[1][0] == 1):
			self.NSRoad.crash()
			self.EWRoad.crash()
			self.total_reward -= -100
			self.end = True
			return -100

		if (NSoutput[0][0] == 1) or (EWoutput[0][0] == 1):
			step_reward += 10 

		wait_penalty = self.wait_weight * (self.NSRoad.totalWait() + self.EWRoad.totalWait())

		step_reward += (self.keepalive - wait_penalty)
		self.total_reward += step_reward
		return step_reward

