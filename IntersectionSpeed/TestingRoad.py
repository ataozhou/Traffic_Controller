
from ComplexCar import ComplexCar
import random
import numpy as np

class Road:

	def __init__(self, numRoads):
		self.EWRoad = [(0, ComplexCar(0, 0))] * (3 + numRoads)
		self.NSRoad = [(0, ComplexCar(0,0))] * (3 + numRoads)
		self.done = False
		self.totalReward = 0
		self.carPositions = []
		self.numRoads = numRoads
		self.numCars = 0

	def numCars(self):
		return self.numCars

	def putCar(self, road, pos, car):
		if road == 1:
			self.EWRoad[pos] = (1, car)
		else:
			self.NSRoad[pos] = (1, car)

	def binaryRepresentation(self):
		temp = np.array([bit[0] for bit in self.EWRoad])
		temp2 = np.array([bit[0] for bit in self.NSRoad])
		return np.array([temp, temp2])


	def newInstance(self):
		self.EWRoad = [(0, ComplexCar(0,0))] * (3+ self.numRoads)
		self.NSRoad = [(0, ComplexCar(0,0))] * (3 + self.numRoads)
		self.done = False


	def gameEnd(self):
		return self.done


	def getState(self):
		binary = self.binaryRepresentation()
		wait = []
		speed = []
		for i in range(self.numRoads + 3):
			wait.append(self.EWRoad[i][1].wait_time)
			speed.append(self.EWRoad[i][1].speed)
		for i in range(self.numRoads + 3):
			wait.append(self.NSRoad[i][1].wait_time)
			speed.append(self.NSRoad[i][1].speed)

		return np.array([binary, np.array(wait), np.array(speed)])


	def crash(self):
		self.EWRoad = [(-1, ComplexCar(0,0))] * (3 + self.numRoads)
		self.NSRoad = [(-1, ComplexCar(0,0))] * (3 + self.numRoads)

	def updateStep(self, stepNum):
		for i in range(self.numRoads + 3):
			if(self.EWRoad[i][1].speed >= stepNum and self.EWRoad[i][0] == 1):
				cars = self.EWRoad[i - 1][0] + 1
				if(cars == 2):
					return 0
				speed = self.EWRoad[i][1].speed
				wait_time = self.EWRoad[i][1].wait_time
				self.EWRoad[i - 1] = (cars, ComplexCar(wait_time, speed))
				self.EWRoad[i] = (0, ComplexCar(0,0))
		for i in range(self.numRoads + 3):
			if(self.NSRoad[i][1].speed >= stepNum and self.NSRoad[i][0] == 1):
				cars = self.NSRoad[i - 1][0] + 1
				if(cars == 2):
					return 0
				speed = self.NSRoad[i][1].speed
				wait_time = self.NSRoad[i][1].wait_time
				self.NSRoad[i - 1] = (cars, ComplexCar(wait_time, speed))
				self.NSRoad[i] = (0, ComplexCar(0,0))
		if(self.EWRoad[1][0] == 1) and (self.NSRoad[1][0] == 1):
			return 0
		return 1


	def step(self, pos, action):
		self.EWRoad[pos][1].drive(action)
		reward = 0
		wait = self.EWRoad[pos][1].wait_time
		reward += (-2 + -2 * wait)
		output = []
		chance = random.random()
		#list - 1
		finishCar = self.EWRoad.pop(0)
		finishCarNS = self.NSRoad.pop(0)
		self.EWRoad.insert(0, (0,ComplexCar(0,0)))
		self.NSRoad.insert(0, (0,ComplexCar(0,0)))
		if(finishCar[0] == 1):
			done = True
			reward += 40
		update = self.updateStep(1)
		if(update == 0):
			self.crash()
			self.done = True
			self.totalReward = -100
			return reward

		finishCar2 = self.EWRoad.pop(0)
		finishCarNS2 = self.NSRoad.pop(0)
		self.EWRoad.insert(0, (0, ComplexCar(0,0)))
		self.NSRoad.insert(0, (0,ComplexCar(0,0)))
		if(finishCar2[0] == 1):
			done = True
			reward += 40
		update = self.updateStep(2)
		if(update == 0):
			self.crash()
			self.done = True
			self.totalReward = -100
			return reward

		if(self.EWRoad[1][0] == 1) and (self.NSRoad[1][0] == 1):
			self.crash()
			self.done = True
			self.totalReward = -100
			return reward

		'''if self.sections[len(self.sections) - 1][0] == 0:
			if (chance <= self.prob):
				self.sections[len(self.sections) - 1] = (1, ComplexCar(0,2))
				self.numCars += 1'''
		self.totalReward += reward
		return reward




	
  

