class Objects:

	def __init__(self,obj, x, y , otype,startFrame,trackNo):
		self.trackNo = trackNo
		self.obj = obj
		self.x =x
		self.y = y
		self.objType = otype
		self.objectLabel = ''
		self.activationLevel  = ''
		self.startFrame = startFrame
		self.EndFrame = startFrame

	def setEndFrame(self,frameNo):
		self.EndFrame = frameNo

	def setObjectLabel (self,label):
		self.objectLabel = label

	def getObjectType(self):
		return self.objType 

	def getObject(self):
		return self.obj

	def setActivationLevel(self,prob):
		self.activationLevel = prob

	def getActivationLevel(self):
		return self.activationLevel



