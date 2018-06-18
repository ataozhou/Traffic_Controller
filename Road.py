
from Car import Car
import random
import numpy as np

class Road:

	def __init__(self, length, prob):
		self.sections = [(0, Car())] * (length + 2)
		self.prob = prob
		self.length = length

	def binaryRepresentation(self):
		temp = np.array([bit[0] for bit in self.sections])
		return temp

	def totalWait(self):
		total_wait = sum([car[1].wait_time for car in self.sections])
		return total_wait

	def step(self,wait):
		chance = random.random()
		#list - 1
		output = self.sections.pop(0)

		#successful wait command
		if wait and (self.sections[1][0] == 1):
			print("waiting")
			#list + 1
			self.sections.insert(1, (0,Car()))
			self.sections[2][1].wait()
			#list -1 or -0
			for i in range(3, self.length + 1):
				if self.sections[i][0] == 0:
					self.sections.pop(i)
					break
				else:
					self.sections[i][1].wait()

		if (self.length + 2) != len(self.sections):
			if (chance <= self.prob):
				self.sections.append([1, Car()])
			else:
				self.sections.append([0, Car()])
		self.length = len(self.sections) - 2
		return (output, self.sections[1])

	def crash(self):
		self.sections = [(-1, Car())] * (self.length + 2)


