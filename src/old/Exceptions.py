class Error(Exception):
	#Base class for excpetions
	def __init__(self, msg):
		self.msg = msg;

	#returns error message
	def get_msg(self):
		return self.msg

class AudioStreamException(Error):
	"""Rasised when an invalid audio stream operation is taken

	Attributes:
		expr -- invalid expression that caused the error
		msg -- explination of the error
	"""
	def __init__(self, msg, expr):
		super().__init__(msg)
		self.expr = expr

	#returns msg
	def get_msg(self):
		return self.msg

	#returns expr
	def get_expr(self):
		return self.expr

	def __str__(self):
		return ("invalid expression: "
				+ self.get_expr()
				+ " | "
				+ self.get_msg()
				)

class DataBridgeException(Error):
	"""Rasised when an invalid audio stream operation is taken

	Attributes:
		expr -- invalid expression that caused the error
		msg -- explination of the error
	"""
	def __init__(self, msg, expr):
		super().__init__(msg)
		self.expr = expr

	#returns msg
	def get_msg(self):
		return self.msg

	#returns expr
	def get_expr(self):
		return self.expr

	def __str__(self):
		return ("invalid expression: "
				+ self.get_expr()
				+ " | "
				+ self.get_msg()
				)
