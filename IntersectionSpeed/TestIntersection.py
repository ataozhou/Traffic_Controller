
from ComplexCar import ComplexCar
from Road import Road
import numpy as np



class Intersection:

	def __init__(self, numRoads, length, prob, keepalive, wait_weight):
		self.NSRoads = [Road(length, numRoads, prob)] * numRoads
		self.EWRoads = [Road(length, numRoads, prob)] * numRoads
		self.intersection = [[0]*numRoads]*numRoads
		self.keepalive = keepalive
		self.total_reward = 0
		self.numRoads = numRoads
		self.length = length
		self.prob = prob
		self.end = False
		self.wait_weight = wait_weight
		self.numSteps = 0

	def numCars(self):
		numCars = 0
		for i in range(self.numRoads):
			numCars += NSRoads[i].numCars() + EWRoads[i].numCars()

	def getState(self):
		state = []
		for i in range(self.numRoads):
			state.append(self.NSRoads[i].getState())
			state.append(self.EWRoad[i].getState())
		return np.array(state)

	def checkIntersection(self):
		for i in range(1, 1+self.numRoads):
			for j in range(1, 1 + self.numRoads):
				self.intersection[i-1][j-1] = NSRoads[j-1][i][0] + EWRoads[i-1][j][0]


	def getBinary(self):
		NSBinary = []
		EWBinary = []
		for road in self.NSRoads:
			NSBinary.append(road.getBinary())
		for road in self.EWRoads:
			EWBinary.append(road.getBinary())

		return np.array([NSBinary, EWBinary])

	def newInstance(self):
		for road in NSRoads:
			road.newInstance()
		for road in EWRoads:
			road.newInstance()
		self.end = False
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
			print ("NS is waiting")
		if(EW_action == 1):
			print("EW is waiting")'''
		#self.numSteps += 1
		NSoutput = self.NSRoad.simpleStep(NS_action)
		EWoutput = self.EWRoad.simpleStep(EW_action)
		step_reward = 0

		if ((NSoutput[1][0] == 1) and (EWoutput[1][0] == 1)):
			self.NSRoad.crash()
			self.EWRoad.crash()
			self.total_reward -= 100
			self.end = True
			return -100

		if (NSoutput[0][0] == 1):
			step_reward += 20

		if (EWoutput[0][0] == 1):
			step_reward += 20

		wait_penalty = self.wait_weight * (self.NSRoad.totalWait() + self.EWRoad.totalWait())

		step_reward += (self.keepalive - wait_penalty)
		self.total_reward += step_reward
		self.setIntersection()
		return step_reward

