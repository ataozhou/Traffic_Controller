
class ComplexCar:

	def __init__(self, wait_time, speed):
		self.wait_time = wait_time
		self.speed = speed

	def wait(self, wait):
		if wait == 0:
			self.wait_time += 0.25

	def copy(self):
		copy = ComplexCar(self.wait_time, self.speed)
		return copy