import numpy as np

#TODO Create data structur Wheel

class Wheel:
	def __init__(self, values):
		self.pos = 0
		self.values = np.array(values)

	def get_length(self):
		return len(self.values)

	def reset_pos(self):
		self.pos = 0

	def get_values(self, i):
		return self.values[i]

	def get_rot_values(self, num):
		i = np.arange(self.pos, self.pos+num)
		l = self.get_length()
		return i % l

	def next_value(self, num=1):
		i = self.get_rot_values(num)
		values = self.get_values(i)
		self.pos += 1
		return values.tolist()

class ColorWheel(Wheel):
	def __init__(self, colors):
		super().__init__(colors)
		self.BLACK = [[0, 0, 0]]
		self.WHITE = [[255, 255, 255]]

class LVWheel(Wheel):
	def __init__(self, lvs):
		super().__init__(lvs)
