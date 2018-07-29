import numpy as np

class Color:
	def __init__(self, color_depth):
		self.rgb_rnbw_cos = []
		self.calc_rgb_rnbw_cos(color_depth)
		
	
	def calc_rgb_rnbw_cos(self, color_depth):
		x = []
		for i in range(0, color_depth):
			x.append(i)
		r = []
		for i in range(0, color_depth):
			r.append((np.cos(i*(2*np.pi)/(color_depth - 1))+1)*(0.5*255))
		g = []
		for i in range(0, color_depth):
			g.append((np.cos((i-np.pi)*(2*np.pi)/(color_depth - 1))+1)*(0.5*255))
		b = []
		for i in range(0, color_depth):
			b.append((np.cos((i+np.pi)*(2*np.pi)/(color_depth - 1))+1)*(0.5*255))
		for i in range(0,color_depth):
			self.rgb_rnbw_cos.append([r[i], g[i], b[i]])
	
	
	def get_rgb_rnbw_cos(self, x, lv):
		return self.rgb_rnbw_cos[x] + lv
		
		
