
from Car import Car
from Road import Road
import numpy as np



class Intersection:

	def __init__(self, length, prob, keepalive, wait_weight):
		self.NSRoad = Road(length,prob)
		self.EWRoad = Road(length,prob)
		self.passedCars = 0
		self.keepalive = keepalive
		self.total_reward = 0
		self.length = length
		self.prob = prob
		self.end = False
		self.wait_weight = wait_weight
		self.numSteps = 0

	def getState(self):
		temp = np.array([self.NSRoad.binaryRepresentation(), self.NSRoad.carRepresentation(), self.EWRoad.binaryRepresentation(), self.EWRoad.carRepresentation()])
		return temp

	def getBinary(self):
		state = []
		temp = self.NSRoad.binaryRepresentation()
		for each in temp:
			state.append(each)
		state.append(self.NSRoad.totalWait())
		temp = self.EWRoad.binaryRepresentation()
		for each in temp:
			state.append(each)
		state.append(self.NSRoad.totalWait())
		state.append(self.passedCars)
		return np.array(state)

	def newInstance(self):
		self.NSRoad = Road(self.length, self.prob)
		self.EWRoad = Road(self.length, self.prob)
		self.end = False
		self.passedCars = 0
		self.total_reward = 0

	def getReward(self):
		return self.total_reward

	def gameEnd(self):
		return self.end

	def gameWait(self):
		print("Wait NS: " + str(self.NSRoad.totalWait()))
		print("Wait EW: " + str(self.EWRoad.totalWait()))

	def step(self,NS_action, EW_action):
		'''if(NS_action == 1):
			self.NSwait += 1
		else:
			self.NSwait = 0
		if(EW_action == 1):
			self.EWwait += 1
		else:
			self.EWwait = 0'''
		#self.numSteps += 1
		NSoutput = self.NSRoad.step(NS_action)
		EWoutput = self.EWRoad.step(EW_action)
		step_reward = 0

		if ((NSoutput[1][0] == 1) and (EWoutput[1][0] == 1)):
			self.NSRoad.crash()
			self.EWRoad.crash()
			self.total_reward -= 100
			self.end = True
			return -100

		if (NSoutput[0][0] == 1):
			step_reward += 20
			self.passedCars +=1

		if (EWoutput[0][0] == 1):
			step_reward += 20
			self.passedCars +=1

		if (self.passedCars >= 20):
			self.NSRoad.win()
			self.EWRoad.win()
			self.total_reward += 100
			self.end = True
			return 100
		wait_penalty = self.wait_weight * (self.NSRoad.totalWait() + self.EWRoad.totalWait())

		step_reward += (self.keepalive - wait_penalty)
		self.total_reward += step_reward
		return step_reward
