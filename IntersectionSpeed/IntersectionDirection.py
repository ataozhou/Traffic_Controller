
from ComplexCar import ComplexCar
from RoadDirection import Road
import numpy as np
import random



class Intersection:

	def __init__(self, numRoads, length, prob, speed, keepalive, wait_weight):
		self.NSRoads = [Road(length, numRoads, prob, speed) for i in range(numRoads)]
		self.EWRoads = [Road(length, numRoads, prob, speed) for i in range(numRoads)]
		self.intersection = [0]*(numRoads + 2)
		for i in range(len(self.intersection)):
			self.intersection[i] = [0]* (numRoads + 2)
		self.keepalive = keepalive
		self.total_reward = 0
		self.numRoads = numRoads
		self.length = length
		self.prob = prob
		self.speed = speed
		self.end = False
		self.wait_weight = wait_weight
		self.numSteps = 1


	def trafficSaturation(self):
		totalCars = 0.0
		for i in range(self.numRoads):
			totalCars += self.NSRoads[i].numCars + self.EWRoads[i].numCars
		trafficPerc = float(totalCars/((self.length* 2) + self.numRoads))
		if trafficPerc > 1:
			print("Traffic Overload: " + str(trafficPerc) + " = " + str(totalCars) + "/ " + str(self.length + self.numRoads + 1) )
		return trafficPerc


	def chooseRoad3(self):
		intersectionNS = 0
		intersectionEW = 0
		totalNSCars = 0
		totalEWCars = 0
		for i in range(self.numRoads):
			intersectionNS += self.NSRoads[i].sections[self.numRoads + 1][0]
			intersectionEW += self.EWRoads[i].sections[self.numRoads + 1][0]
		if intersectionNS > intersectionEW:
			return [1, 0]
		elif intersectionEW > intersectionNS:
			return [0, 1]
		else:
			for i in range(self.numRoads):
				totalNSCars += self.NSRoads[i].numCars
				totalEWCars += self.EWRoads[i].numCars

			if(totalNSCars > totalEWCars):
				return [1, 0]
			else:
				return[0,1]


	def chooseRoad2(self):
		intersectionNS = 0
		intersectionEW = 0
		for i in range(self.numRoads):
			intersectionNS += self.NSRoads[i].sections[self.numRoads + 1][0]
			intersectionEW += self.EWRoads[i].sections[self.numRoads + 1][0]
		if intersectionNS > intersectionEW:
			return [1, 0]
		elif intersectionEW > intersectionNS:
			return [0, 1]
		else:
			chance = random.random()
			if chance < 0.5:
				return [1, 0]
			else:
				return [0, 1]



	def getState(self):
		state = []
		wait = []
		for i in range(self.numRoads):
			for j in range(len(self.NSRoads[i].sections)):
				state.append(self.NSRoads[i].sections[j][0])
				state.append(self.NSRoads[i].sections[j][1].wait_time)

		for i in range(self.numRoads):
			for j in range(len(self.EWRoads[i].sections)):
				state.append(self.EWRoads[i].sections[j][0])
				state.append(self.EWRoads[i].sections[j][1].wait_time)


		return np.array(state).ravel()



	def getSimpleState(self):
		state = []
		wait = []
		for i in range(self.numRoads):
			for j in range(5):
				state.append(self.NSRoads[i].sections[j][0])
				state.append(self.NSRoads[i].sections[j][1].wait_time)

		for i in range(self.numRoads):
			for j in range(5):
				state.append(self.EWRoads[i].sections[j][0])
				state.append(self.EWRoads[i].sections[j][1].wait_time)


		return np.array(state).ravel()



	def getSimpleBinary(self):
		temp = []
		for i in range(self.numRoads):
			temp.append(self.NSRoads[i].getSimpleBinary())
			temp.append(self.EWRoads[i].getSimpleBinary())
		return np.array(temp)


	def getIntersection(self):
		return np.array(self.intersection)


	def windowUpdate(self):
		EWIntro = []
		NSIntro = []
		for i in range(0, self.numRoads):
			NSIntro.append(i, self.NSRoads.carIntroduction())
			EWIntro.append(i, self.EWRoads.carIntroduction())

			


	def binaryRepresentation(self):
		tempNS = []
		tempEW = []
		for i in range(self.numRoads):
			tempNS.append(self.NSRoads[i].binaryRepresentation())
			tempEW.append(self.EWRoads[i].binaryRepresentation())
		return np.array([tempNS, tempEW])



	def setIntersection(self):
		self.intersection = [0]*(self.numRoads + 2)
		for i in range(len(self.intersection)):
			self.intersection[i] = [0]* (self.numRoads + 2)
		end = self.numRoads + 1
		roadEnd = self.numRoads + 1
		for i in range(1, self.numRoads + 1):
			for j in range(1, self.numRoads + 1):
				value =  self.NSRoads[j-1].sections[i][0] + self.EWRoads[i-1].sections[j][0]
				self.intersection[i][j] = value
				#print("Intersection[" + str(i) + "][" + str(j) + "] = " + str(value))
				if(value > 1):
					return -1


		for i in range(1, self.numRoads + 1):
			self.intersection[i][0] = self.EWRoads[i - 1].sections[0][0]
			self.intersection[i][end] = self.EWRoads[ i - 1].sections[roadEnd][0]

			self.intersection[0][i] = self.NSRoads[i - 1].sections[0][0]
			self.intersection[end][i] = self.NSRoads[i - 1].sections[roadEnd][0]


		'''self.intersection[2][0] = self.EWRoads[1].sections[0][0]
		self.intersection[2][end] = self.EWRoads[1].sections[roadEnd][0]
		self.intersection[3][0] = self.EWRoads[2].sections[0][0]
		self.intersection[3][end] = self.EWRoads[2].sections[roadEnd][0]'''

		'''self.intersection[0][2] = self.NSRoads[1].sections[0][0]
		self.intersection[end][2] = self.NSRoads[1].sections[roadEnd][0]
		self.intersection[0][3] = self.NSRoads[2].sections[0][0]
		self.intersection[end][3] = self.NSRoads[2].sections[roadEnd][0]'''

		

		


		for i in range(1, self.numRoads + 1):
			#print(self.intersection[i])
			self.EWRoads[i-1].setIntersection(self.intersection[i])

		for i in range(1, self.numRoads + 1):
			temp = []
			for j in range(0, self.numRoads + 2):
				temp.append(self.intersection[j][i])
			#print(temp)
			self.NSRoads[i-1].setIntersection(temp)

		#print(np.array(self.intersection))
		
		return 1


	def getBinary(self):
		NSBinary = []
		EWBinary = []
		for road in self.NSRoads:
			NSBinary.append(road.binaryRepresentation())
		for road in self.EWRoads:
			EWBinary.append(road.binaryRepresentation())

		return np.array([NSBinary, EWBinary])

	def newInstance(self):
		for road in self.NSRoads:
			road.newInstance(self.prob)
		for road in self.EWRoads:
			road.newInstance(self.prob)
		self.end = False
		self.total_reward = 0




	def getReward(self):
		return self.total_reward

	def getWait(self):
		wait = 0
		for i in range(self.numRoads):
			wait += self.EWRoads[i].totalWait() + self.NSRoads[i].totalWait()
		return wait

	def gameEnd(self):
		return self.end


	def trainstep(self, NSaction, EWaction):
		self.numSteps += 1

		'''if(self.numSteps == 100):
			self.numSteps = 1
			for i in range(self.numRoads):
				self.NSRoads[i].updateProb(3)
				self.EWRoads[i].updateProb(3)'''
				
		step_reward = 0
		wait = self.getWait()
		passedCars = 0
		crash = self.setIntersection()
		output = []
		if crash == -1:
			self.NSRoads[0].crash()
			self.EWRoads[0].crash()
			self.total_reward -= 100
			self.end = True
			return -100

		if self.speed > 1:
			for i in range(self.numRoads):
				output.append(self.NSRoads[i].updateStep(NSaction, 0))
				crash = self.setIntersection()
			for i in range(self.numRoads):
				output.append(self.EWRoads[i].updateStep(EWaction, 0))
				crash = self.setIntersection()

		if(NSaction == 1):
			for i in range(self.numRoads):
				output.append(self.EWRoads[i].step(0))
				crash = self.setIntersection()

			for i in range(self.numRoads):
				output.append(self.NSRoads[i].step(1))
				crash = self.setIntersection()


		elif(EWaction == 1):
			for i in range(self.numRoads):
				output.append(self.NSRoads[i].step(0))
				crash = self.setIntersection()

			for i in range(self.numRoads):
				output.append(self.EWRoads[i].step(1))
				crash = self.setIntersection()

		else:
			for i in range(self.numRoads):
				output.append(self.NSRoads[i].step(0))
				crash = self.setIntersection()
			for i in range(self.numRoads):
				output.append(self.EWRoads[i].step(0))
				crash = self.setIntersection()


		crash = self.setIntersection()
		if crash == -1:
			self.NSRoads[0].crash()
			self.EWRoads[0].crash()
			self.total_reward -= 100
			self.end = True
			return -100

		for i in range(0, len(output)):
			for j in range (0, len(output[i])):
				if(output[i][j]== 1):
					passedCars += 1

		#print("PassedCars: " + str(passedCars))

		if(self.total_reward < -1000):
			for i in range(self.numRoads):
				self.NSRoads[i].crash()
				self.EWRoads[i].crash()
			self.total_reward -= 100
			self.end = True

		wait_penalty = self.wait_weight * (self.getWait())

		step_reward += (-1 * wait_penalty) + (30 * passedCars)
		self.total_reward += step_reward
		#self.setIntersection()
		return step_reward




	def step(self, NSaction, EWaction):
		self.numSteps += 1

		if(self.numSteps == 500):
			self.numSteps = 1
			for i in range(self.numRoads):
				self.NSRoads[i].updateProb(1)
				self.EWRoads[i].updateProb(1)

		step_reward = 0
		wait = self.getWait()
		passedCars = 0
		crash = self.setIntersection()
		output = []
		if crash == -1:
			self.NSRoads[0].crash()
			self.EWRoads[0].crash()
			self.total_reward -= 100
			self.end = True
			return -100

		if self.speed > 1:
			for i in range(self.numRoads):
				output.append(self.NSRoads[i].updateStep(NSaction, 0))
				crash = self.setIntersection()
			for i in range(self.numRoads):
				output.append(self.EWRoads[i].updateStep(EWaction, 0))
				crash = self.setIntersection()

		if(NSaction == 1):
			for i in range(self.numRoads):
				output.append(self.EWRoads[i].step(0))
				crash = self.setIntersection()

			for i in range(self.numRoads):
				output.append(self.NSRoads[i].step(1))
				crash = self.setIntersection()


		elif(EWaction == 1):
			for i in range(self.numRoads):
				output.append(self.NSRoads[i].step(0))
				crash = self.setIntersection()

			for i in range(self.numRoads):
				output.append(self.EWRoads[i].step(1))
				crash = self.setIntersection()

		else:
			for i in range(self.numRoads):
				output.append(self.NSRoads[i].step(0))
				crash = self.setIntersection()
			for i in range(self.numRoads):
				output.append(self.EWRoads[i].step(0))
				crash = self.setIntersection()


		crash = self.setIntersection()
		if crash == -1:
			self.NSRoads[0].crash()
			self.EWRoads[0].crash()
			self.total_reward -= 100
			self.end = True
			return -100

		for i in range(0, len(output)):
			for j in range (0, len(output[i])):
				if(output[i][j]== 1):
					passedCars += 1

		#print("PassedCars: " + str(passedCars))

		'''if(self.total_reward < -1000):
			for i in range(self.numRoads):
				self.NSRoads[i].crash()
				self.EWRoads[i].crash()
			self.total_reward -= 100
			self.end = True'''

		wait_penalty = self.wait_weight * (self.getWait())

		step_reward += (-1 * wait_penalty) + (30 * passedCars)
		self.total_reward += step_reward
		#self.setIntersection()
		return step_reward

