
from ComplexCar import ComplexCar
import random
import numpy as np

class Road:

	def __init__(self, length, numRoads, prob, speed):
		self.sections = [(0, ComplexCar(0,speed))] * (length + 1 + numRoads)
		self.intersectionView = []
		self.intersection = []
		self.roadPosition = []
		#self.actionList = []
		self.numRoads = numRoads
		self.speed = speed
		self.prob = prob
		self.length = length
		self.carWait = 0
		self.numCars = 0


	def binaryRepresentation(self):
		temp = np.array([bit[0] for bit in self.sections])
		return temp

	def updateProb(self, increment):
		self.prob += (0.1 * increment)
		#print(self.prob)
		if self.prob >= 0.9:
			self.prob = 0.1

	def setIntersection(self, intersection):
		self.intersectionView = []
		self.intersection = intersection
		for i in range(len(intersection)):
			if(intersection[i] == 1):
				self.intersectionView.append(i)
		self.intersectionView.sort()

		'''print(len(self.intersectionView))
		print(self.intersectionView)
		print(self.intersection)'''

	def carIntroduction(self):
		output = self.numCars
		self.numCars = 0
		return output


	'''def actionList(self):
		output = self.actionList
		self.actionList = []
		return output'''

	def newInstance(self, prob):
		self.sections = [(0, ComplexCar(0, self.speed))] * (self.length + 1 + self.numRoads)
		self.prob = prob
		self.numCars = 0
		self.intersection = []
		self.intersectionView = []
		self.roadPosition = []
		self.carWait = 0


	def getState(self):
		binary = []
		wait = []

		for i in range(len(self.sections)):
			binary.append(self.sections[i][0])
			wait.append(self.sections[i][1].wait_time)

		state = binary + wait
		return np.array(state)

	def getSimpleBinary(self):
		state = []
		for i in range(5):
			state.append(self.sections[i][0])
		return state

	def crash(self):
		self.sections = [(-1, ComplexCar(0, self.speed))] * (self.length + 1 + self.numRoads)


	def updatePos(self):
		self.roadPosition = []
		for i in range(self.numRoads + 2, len(self.sections)):
			if(self.sections[i][0] == 1):
				self.roadPosition.append(i)
		self.roadPosition.sort()

	def updateStep(self, action, wait):
		output = []
		for j in range(0, len(self.intersectionView)):
			index = self.intersectionView[j]
			if(action == 0) and (index == (self.numRoads + 1)):
				self.sections[index][1].wait(0)
				break

			if(self.sections[index][0] == 1):
				i = 1
				for i in range(1, self.speed + 1):
					if(index - i < 0) or (self.intersection[index-i] == 1):
						i -= 1
						self.carWait += 0
						break
				if(wait == 1):
					self.sections[index][1].wait(self.carWait + i)
					#self.actionList.append(self.carWait)
					self.carWait = 0
				else:
					self.carWait +=1
				wait_time = self.sections[index][1].wait_time
				self.sections[index - i] = (1, ComplexCar(wait_time,self.speed))
				self.intersection[index - i] = 1
				if i != 0:
					self.sections[index] = (0, ComplexCar(0, self.speed))
					self.intersection[index] = 0
				if (index - i) == 0:
					popCar = self.sections.pop(0)
					if popCar[0] == 1:
						#print("Here2")
						output.append(popCar[0])
						self.numCars -= 1
					self.sections.insert(0, (0, ComplexCar(0, self.speed)))
		return output


	def totalWait(self):
		total_wait = sum([car[1].wait_time for car in self.sections])
		return total_wait


	def step(self, action):
		self.updatePos()
		output = []
		chance = random.random()
		#list - 1
		finishCar = self.sections.pop(0)
		if(finishCar[0] == 1):
			output.append(finishCar[0])
		self.sections.insert(0, (0,ComplexCar(0, self.speed)))
		update = []
		if(action == 1):
			update = self.updateStep(1,1)
		else:
			update = self.updateStep(0,1)

		'''if(self.sections[self.numRoads + 2][0] == 1) and (self.sections[self.numRoads + 1][0] != 1):
			self.sections[self.numRoads + 1] = (1, ComplexCar(self.sections[self.numRoads +2][1].wait_time, self.speed))
			self.sections[self.numRoads + 2] = (0, ComplexCar(0, self.speed))'''

		for j in range(0, len(self.roadPosition)):
			index = self.roadPosition[j]
			#print(index)
			i = 1
			for i in range(1, self.speed + 1):
				if(index - i < self.numRoads + 1) or (self.sections[index - i][0] != 0):
					#print("here")
					i -= 1
					break

			self.sections[index][1].wait(i)
			#self.actionList.append(i)
			wait_time = self.sections[index][1].wait_time
			self.sections[index - i] = (1, ComplexCar(wait_time, self.speed))
			if i != 0:
				self.sections[index] = (0, ComplexCar(0, self.speed))
		
		if self.sections[len(self.sections) - 1][0] == 0:
			if (chance <= self.prob):
				self.sections[len(self.sections) - 1] = (1, ComplexCar(0, self.speed))
				self.numCars += 1

		return np.array([output + update])

