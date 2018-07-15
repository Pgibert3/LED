class color_wheel:
	def __init__(self, colors):
		self.colors = colors
		self.pos = 0
	
	
	def get_color(self):
		return self.colors[self.pos]
	
	
	def reset_colors(self, colors):
		self.colors = colors
	
	
	def reset_pos(self):
		self.pos = 0
	
	
	def reset(self, colors):
		reset_colors(colors)
		reset_pos()
	
	
	def set_color(self, i, color):
		self.colors[i] = color
	
	
	def rotate(self, step):
		self.pos = (self.pos + (step % len(self.colors))) % len(self.colors)
	
	
	
	
	
	
