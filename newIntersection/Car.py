
class Car:

	def __init__(self, wait_time, speed):
		self.wait_time = wait_time
		self.speed = speed

	def drive(self, action):
		if action == 1:
			#print("speed up")
			if self.speed < 1:
				self.speed += 1
		elif action == -1:
			#print("slow down")
			if self.speed > 0:
				self.speed -= 1
		self.wait()
	def wait(self):
		#if self.speed == 1:
			#self.wait_time += 0.5
		if self.speed == 0:
			self.wait_time += 1

