import numpy as np
import cv2

import sys
import math

from vgg16 import vgg16
import tensorflow as tf
from imagenet_classes import class_names

class StillObjects:

	def __init__(self):
		self.oldList = []

	def isPointSelected(self,keyPoint):
		x1 = keyPoint.pt[0]
		y1 = keyPoint.pt[1]
		isMatch = False
		for p in self.oldList:
			x2 = p.pt[0]
			y2 = p.pt[1]
			dist = math.hypot(x2 - x1, y2 - y1)
			ratio = dist/(keyPoint.size/2 + p.size/2)
			if ratio <= 0.5:
				isMatch = True
				break

		return isMatch

	def distance(self,p1, p2):
		x1 = p1.pt[0]
		y1 = p1.pt[1]
		x2 = p2.pt[0]
		y2 = p2.pt[1]
		dist = math.hypot(x2 - x1, y2 - y1)
		return dist

	def getCentroid(self,pts, radius):

		x = 0
		y = 0
		a=0
		for p in pts:
			x+= p.pt[0]
			y+= p.pt[1]
			a+= p.angle

		x = x/len(pts)
		y = y/len(pts)
		a = a/len(pts)

		return cv2.KeyPoint(x,y,2*radius,a)


	def getPointsinRadius(self,kp,radius):

		points=[]

		selectedList = []
		for p in kp:
			if not p in selectedList:
				nearby  = []
				nearby = filter(lambda x: self.distance(x,p) <= radius, kp)
				pt = self.getCentroid(nearby,radius)
				points.append(pt)

		return points

	#height, width = img1.shape[:2]
	#res = cv2.resize(img1,(width//2, height//2), interpolation = cv2.INTER_AREA)
	def getStillObject(self,frame):

		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		blur = cv2.medianBlur(gray,5)
		star = cv2.FeatureDetector_create('SIFT')
		kp = star.detect(blur)


		maxSize = max(p.size for p in kp)

		thres=100 if maxSize >=100 else 0.8*maxSize

		filterdPoints = filter(lambda p:p.size >= thres, kp)
		smallPoints = filter(lambda p:p.size>=5 and p.size<=thres , kp) 

		rPoints = self.getPointsinRadius(smallPoints,100)

		filterdPoints.extend(rPoints)
		if len(filterdPoints) > 0:
			newlist = sorted(filterdPoints, key=lambda x: x.size,reverse=True)

		for p in filterdPoints:
			if not self.isPointSelected(p):
				tl_X = int(p.pt[0])-int((p.size)//2)
				tl_y = int(p.pt[1])-int((p.size)//2)
				br_X = int(p.pt[0])+int((p.size)//2)
				br_y = int(p.pt[1])+int((p.size)//2)
				
				#cv2.circle(frame,(int(p.pt[0]),int(p.pt[1])), int((p.size)//2+1),(0,0,255),1)
				pts1 = np.float32([[tl_X,tl_y],[tl_X,br_y],[br_X,br_y]])

				pts2 = np.float32([[0,0],[0,224],[224,224]])

				M = cv2.getAffineTransform(pts1,pts2)

				obj = cv2.warpAffine(frame,M,(224,224))

				#cv2.rectangle(frame,(tl_X,tl_y),(br_X,br_y),(0,255,0),3)

				if len(self.oldList) == 30:
					del self.oldList[0]

				self.oldList.append(p)
				# object , topleft , bottom right
				return obj, tl_X,tl_y,br_X,br_y,p
