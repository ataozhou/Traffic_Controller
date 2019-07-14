
from ComplexCar import ComplexCar
import random
import numpy as np

class Road:

	def __init__(self, length, numRoads, prob):
		self.sections = [(0, ComplexCar(0))] * (length + 1 + numRoads)
		self.speed = 2
		self.numRoads = numRoads
		self.carPositions = []
		self.prob = prob
		self.length = length
		self.numCars = 0

	def numCars(self):
		return self.numCars

	def binaryRepresentation(self):
		temp = np.array([bit[0] for bit in self.sections])
		return temp


	def newInstance(self):
		self.sections = [(0, ComplexCar(0))] * (self.length + 1 + self.numRoads)


	def getState(self):
		binary = []
		wait = []
		speed = [self.speed]
		for i in range(len(self.sections)):
			binary.append(self.sections[i][0])
			wait.append(self.sections[i][1].wait_time)

		state = [each for each in binary] + [each for each in wait] + speed
		return np.array(state)

	def getSimpleState(self):
		output = self.binaryRepresentation()
		for each in self.sections:
			output.append(each[1].wait_time)
		output.append(self.speed)
		return output

	def crash(self):
		self.sections = [(-1, ComplexCar(0))] * (self.length + 1 + self.numRoads)

	def setPositions(self):
		self.carPositions = []
		for i in range(len(self.sections)):
			if(self.sections[i][0] == 1):
				self.carPositions.append(i)

	def updateStep(self, stepNum):
		for i in range(len(self.carPositions)):
			index = self.carPositions[i]
			if(self.speed >= stepNum and self.sections[index][0] == 1):
				cars = self.sections[index - 1][0] + 1
				if(cars == 2):
					return 0

				wait_time = self.sections[index][1].wait_time
				self.sections[index - 1] = (cars, ComplexCar(wait_time))
				self.sections[index] = (0, ComplexCar(0))
		self.setPositions()
		return 1


	def totalWait(self):
		total_wait = sum([car[1].wait_time for car in self.sections])
		return total_wait


	def step(self,action, stepNum):
		if action == 1:
			if self.speed < 2:
				self.speed += 1
		elif action == -1:
			if self.speed > 0:
				self.speed -= 1
		output = []
		chance = random.random()
		#list - 1
		finishCar = self.sections.pop(0)
		self.sections.insert(0, (0,ComplexCar(0)))
		update = self.updateStep(stepNum)
		if(update == 0):
			self.crash()
			output.append(-1)
		else:
			output.append(finishCar[0])
		output.append(self.sections[1])

		if self.sections[len(self.sections) - 1][0] == 0:
			if (chance <= self.prob):
				self.sections[len(self.sections) - 1] = (1, ComplexCar(0))
				self.numCars += 1
		return np.array(output)


	def passiveStep(self, stepNum):
		output = []
		chance = random.random()
		#list - 1
		finishCar = self.sections.pop(0)
		self.sections.insert(0, (0,ComplexCar(0)))
		update = self.updateStep(stepNum)
		if(update == 0):
			self.crash()
			output.append(-1)
		else:
			output.append(finishCar[0])
		output.append(self.sections[1])

		if self.sections[len(self.sections) - 1][0] == 0:
			if (chance <= self.prob):
				self.sections[len(self.sections) - 1] = (1, ComplexCar(0))
				self.numCars += 1
		return np.array(output)



