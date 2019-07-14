
from Car import Car
import random
import numpy as np

class Road:

	def __init__(self, numRoads):
		self.EWRoad = [(0, Car(0, 0))] * (2 + numRoads)
		self.NSRoad = [(0, Car(0,0))] * (2 + numRoads)
		self.done = False
		self.totalReward = 0
		self.numRoads = numRoads


	def putCar(self, road, pos, car):
		if road == 1:
			self.EWRoad[pos] = (1, car)
		else:
			self.NSRoad[pos] = (1, car)

	def binaryRepresentation(self):
		temp = np.array([bit[0] for bit in self.EWRoad])
		temp2 = np.array([bit[0] for bit in self.NSRoad])

		return np.array([temp2 , temp]).ravel()


	def newInstance(self):
		self.EWRoad = [(0, Car(0, 0))] * (2 + self.numRoads)
		self.NSRoad = [(0, Car(0,0))] * (2 + self.numRoads)
		chance = random.random()
		speedRand = random.randint(0,1)
		speedControlled = random.randint(0,1)
		self.putCar(1, 2, Car(0, speedControlled))
		self.putCar(0, 2, Car(0, speedRand))
		if(chance < 0.5):
			self.putCar(0,1, Car(0,1))
		self.totalReward = 0
		self.done = False

	def reuseState(self, state):
		for i in range(0, 2 + self.numRoads):
			self.EWRoad[i] = (state[0][i], Car(state[1][i], state[2][i]))
		for i in range(2 + self.numRoads, 2 * (2 + self.numRoads)):
			self.NSRoad[(i - (2 + self.numRoads))] = (state[0][i], Car(state[1][i], state[2][i]))

		self.totalReward = 0
		self.done = False

	'''def updatePos(self):
		for i in range(self.numRoads + 2):
			if self.EWRoad[i][0] == 1:
				self.carPosition = i'''

	def gameEnd(self):
		return self.done

	def getReward(self):
		return self.totalReward


	def getState(self):
		binary = self.binaryRepresentation()
		#binary = [self.NSRoad[1][0], self.NSRoad[2][0]]
		wait = []
		speed = []
		for i in range(0, self.numRoads + 2):
			wait.append(self.EWRoad[i][1].wait_time)
			speed.append(self.EWRoad[i][1].speed)
		for i in range(0, self.numRoads + 2):
			wait.append(self.NSRoad[i][1].wait_time)
			speed.append(self.NSRoad[i][1].speed)

		#return np.array([binary.ravel(), np.array(wait), np.array(speed)])
		#print(binary)
		#print(wait)
		#print(speed)
		return np.array([binary, wait, speed])


	def crash(self):
		self.EWRoad = [(-1, Car(0,0))] * (2 + self.numRoads)
		self.NSRoad = [(-1, Car(0,0))] * (2 + self.numRoads)

	def updateStep(self, stepNum):
		for i in range(self.numRoads + 2):
			if(self.EWRoad[i][1].speed >= stepNum and self.EWRoad[i][0] == 1):
				cars = self.EWRoad[i - 1][0] + 1
				if(cars == 2):
					return 0
				speed = self.EWRoad[i][1].speed
				wait_time = self.EWRoad[i][1].wait_time
				self.EWRoad[i - 1] = (cars, Car(wait_time, speed))
				self.EWRoad[i] = (0, Car(0,0))
		for i in range(self.numRoads + 2):
			if(self.NSRoad[i][1].speed >= stepNum and self.NSRoad[i][0] == 1):
				cars = self.NSRoad[i - 1][0] + 1
				if(cars == 2):
					return 0
				speed = self.NSRoad[i][1].speed
				wait_time = self.NSRoad[i][1].wait_time
				self.NSRoad[i - 1] = (cars, Car(wait_time, speed))
				self.NSRoad[i] = (0, Car(0,0))
		if(self.EWRoad[1][0] == 1) and (self.NSRoad[1][0] == 1):
			return 0
		#self.updatePos()
		return 1


	def step(self, action):

		#print("Action: " + str(action))
		self.EWRoad[2][1].drive(action)
		reward = 0
		output = []
		chance = random.random()
		#list - 1
		finishCar = self.EWRoad.pop(0)
		finishCarNS = self.NSRoad.pop(0)
		self.EWRoad.insert(0, (0,Car(0,0)))
		self.NSRoad.insert(0, (0,Car(0,0)))
		if(finishCar[0] == 1):
			self.done = True
			reward += 40
			self.totalReward += 40
			return reward
		update = self.updateStep(1)
		if(update == 0):
			self.crash()
			self.done = True
			self.totalReward += -100
			reward = -100
			return reward

		finishCar2 = self.EWRoad.pop(0)
		finishCarNS2 = self.NSRoad.pop(0)
		self.EWRoad.insert(0, (0, Car(0,0)))
		self.NSRoad.insert(0, (0,Car(0,0)))
		if(finishCar2[0] == 1):
			self.done = True
			reward += 40
			self.totalReward += 40
			return reward
		update = self.updateStep(2)
		if(update == 0):
			self.crash()
			self.done = True
			self.totalReward += -100
			reward = -100
			return reward

		if((self.EWRoad[1][0] == 1) and (self.NSRoad[1][0] == 1)) or self.totalReward < -100:
			self.crash()
			self.done = True
			self.totalReward += -100
			reward = -100
			return reward

		'''if self.sections[len(self.sections) - 1][0] == 0:
			if (chance <= self.prob):
				self.sections[len(self.sections) - 1] = (1, Car(0,2))
				self.numCars += 1'''

		wait = self.EWRoad[2][1].wait_time
		#print("wait: " + str(wait))
		reward += -2 + (-2 * wait)
		self.totalReward += reward
		#print("step reward" + str(reward))
		#print("total reward" + str(self.totalReward))
		return reward


	
  

