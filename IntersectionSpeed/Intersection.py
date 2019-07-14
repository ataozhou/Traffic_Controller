
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
			state.append(self.EWRoads[i].getState())
		return np.array(state)

	'''def setIntersection(self):
		for i in range(1, 1+self.numRoads):
			for j in range(1, 1 + self.numRoads):
				self.intersection[i-1][j-1] = self.NSRoads[j-1][i][0] + self.EWRoads[i-1][j][0]'''


	def getBinary(self):
		NSBinary = []
		EWBinary = []
		for road in self.NSRoads:
			NSBinary.append(road.getBinary())
		for road in self.EWRoads:
			EWBinary.append(road.getBinary())

		return np.array([NSBinary, EWBinary])

	def newInstance(self):
		for road in self.NSRoads:
			road.newInstance()
		for road in self.EWRoads:
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


	'''idle = [1,0,0,0,0,0,0,0,0]
	slowNS = [0,1,0,0,0,0,0,0,0]
	speedNS = [0,0,1,0,0,0,0,0,0]
	slowEW = [0,0,0,1,0,0,0,0,0]
	speedEW = [0,0,0,0,1,0,0,0,0]
	slowNSspeedEW = [0,0,0,0,0,1,0,0,0]
	speedNSslowEW = [0,0,0,0,0,0,1,0,0]
	speedBoth = [0,0,0,0,0,0,0,1,0]
	slowBoth = [0,0,0,0,0,0,0,0,1]'''


	def step(self,action):
		index = np.argmax(action)
		NS_action = 0
		EW_action = 0
		if(index == 1):
			NS_action = -1
		if(index == 2):
			NS_action = 1
		if(index == 3):
			EW_action = -1
		if(index == 4):
			EW_action = 1
		if(index == 5):
			NS_action = -1
			EW_action = 1
		if(index == 6):
			NS_action = 1
			EW_action = -1
		if(index == 7):
			NS_action = 1
			EW_action = 1
		if(index == 8):
			NS_action = -1
			EW_action = -1
		NSoutput = self.NSRoads[0].step(NS_action, 1)
		EWoutput = self.EWRoads[0].step(EW_action, 1)
		missingFrame = self.getState()
		
		step_reward = 0

		if ((NSoutput[1][0] == 1) and (EWoutput[1][0] == 1)):
			self.NSRoads[0].crash()
			self.EWRoads[0].crash()
			self.total_reward -= 100
			self.end = True
			return -100, missingFrame, missingFrame

		if (NSoutput[0] == 1):
			step_reward += 20

		if (EWoutput[0] == 1):
			step_reward += 20

		NSpassiveoutput = self.NSRoads[0].passiveStep(2)
		EWpassiveoutput = self.EWRoads[0].passiveStep(2)
		missingFrame2 = self.getState()


		if ((NSpassiveoutput[1][0] == 1) and (EWpassiveoutput[1][0] == 1)):
			self.NSRoads[0].crash()
			self.EWRoads[0].crash()
			self.total_reward -= 100
			self.end = True
			return -100, missingFrame, missingFrame2

		if (NSpassiveoutput[0] == 1):
			step_reward += 20

		if (EWpassiveoutput[0] == 1):
			step_reward += 20

		wait_penalty = self.wait_weight * (self.NSRoads[0].totalWait() + self.EWRoads[0].totalWait())

		step_reward += (self.keepalive - wait_penalty)
		self.total_reward += step_reward
		#self.setIntersection()
		return step_reward, missingFrame, missingFrame2

