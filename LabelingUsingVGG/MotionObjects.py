import cv2
import numpy as np
from matplotlib import pyplot as plt

class MotionObjects:

	def __init__(self):
		self.objectList = []
		
	def doOverlap(self,tl1, br1, tl2, br2,thres):

		tr1 = [br1[0],tl1[1]]
		bl1 = [tl1[0], br1[1]]

		#print(tl1,tr1,bl1,br1)
		tlExists = self.IsPointInRect(tl1, [[tl2[0]-thres,tl2[1]-thres],[br2[0]+thres,br2[1]+thres]] )
		brExists = self.IsPointInRect(br1, [[tl2[0]-thres,tl2[1]-thres],[br2[0]+thres,br2[1]+thres]] )
		blExists = self.IsPointInRect(bl1, [[tl2[0]-thres,tl2[1]-thres],[br2[0]+thres,br2[1]+thres]])
		trExists= self.IsPointInRect(tr1, [[tl2[0]-thres,tl2[1]-thres],[br2[0]+thres,br2[1]+thres]] )

		if tlExists or brExists or blExists or trExists:
			isOverLap = True
		else:
			isOverLap = False



		if not isOverLap:

			tr2 = [br2[0],tl2[1]]
			bl2 = [tl2[0], br2[1]]

			#print(tl2,tr2,bl2,br2)
			#print('----------------------------------------------------')
			tlExists = self.IsPointInRect(tl2, [[tl1[0]-thres,tl1[1]-thres],[br1[0]+thres,br1[1]+thres]] )
			brExists = self.IsPointInRect(br2, [[tl1[0]-thres,tl1[1]-thres],[br1[0]+thres,br1[1]+thres]] )
			blExists = self.IsPointInRect(bl2, [[tl1[0]-thres,tl1[1]-thres],[br1[0]+thres,br1[1]+thres]])
			trExists = self.IsPointInRect(tr2, [[tl1[0]-thres,tl1[1]-thres],[br1[0]+thres,br1[1]+thres]] )

			if tlExists or brExists or blExists or trExists:
				isOverLap = True
			else:
				isOverLap = False
		else:
			return isOverLap

		return isOverLap

	def IsPointInRect(self,p,rect):

		tl = rect[0]
		br = rect[1]
		#print(p,tl,br)

		if p[0] >= tl[0] and p[1]>=tl[1]:
			if p[0] <=br[0] and p[1] <= br[1]:
				return True
		return False



	def getRectangles2(self,rect,thres):

		#print('rect = {} len(rect) = {}'.format(rect, len(rect)))
		
		maps = dict()
		count = 0
		for i in range(len(rect)):
			if len(maps) == 0:
				maps[count] = rect[i]
				count+=1
			else:
				x1,y1 = rect[i][0],rect[i][1]				
				w,h = rect[i][2],rect[i][3]

				x2,y2 = x1+w, y1+h

				'''x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
				x, y = int(x1-0.5*w), int(y1-0.5*h)
				x2,y2 = int(x1+0.5*w),int(y1+0.5*h)'''
				isOverLap = False
				
				for key in list(maps):
					
					rect1 = maps[key]
					tl = [rect1[0],rect1[1]]
					br = [rect1[0]+rect1[2],rect1[1]+rect1[3]]
					isOverLap = self.doOverlap([x1,y1],[x2,y2],tl,br,thres)
					if isOverLap:
						a = [x1,y1]
						b = [x2,y2]
						point = np.asarray([a,b,tl,br])
						#print('point = {}', point)
						#new_rect = cv2.boundingRect(point)
						mina = np.min(point,axis=0)
						maxa = np.max(point,axis=0)
						maps[key] = (mina[0],mina[1],maxa[0]-mina[0], maxa[1]-mina[1])
						break
				if not isOverLap:
					maps[count] = rect[i]
					count+=1
				#print('maps = {}'.format(maps))
		rectList =[]
		rectList = maps.values()
		
		return rectList

	def getObjects(self,frameMask, current_frame,minObjThreshold=100, maxObjThreshold=9999, niterations = 5,matchMargin=0.09):

		kernel = np.ones((9, 9),np.uint8)

		newMask = cv2.dilate(frameMask,kernel,iterations =niterations)

		cv2.imshow('mask',newMask)
		(contours, _) = cv2.findContours(newMask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE) 

		#print("contours length ={}  ".format(len(contours)))

		newList = []

		#cv2.rectangle (current_frame,(0,frameMask.shape[0]//8),(frameMask.shape[1], 2*frameMask.shape[0]//3),(0,0,255),1)

		#print(topLeft,bottomRight)
		maxSize=0
		if len(contours) >0:
			maxSize=max(x.shape[0] for x in contours)
			#print('maxSize = ',maxSize)

		finalObjectsLoc = []
		finalObjects=[]

		if maxSize >= minObjThreshold and maxSize <= maxObjThreshold:
			for l in contours:
				if l.shape[0]>= minObjThreshold and l.shape[0] <=maxSize:
					p = l.reshape(l.shape[0],l.shape[2])
					min = np.amin(p,axis=0)
					maxp = np.amax(p,axis=0)

					r = cv2.boundingRect(l)
					newList.append(r)

		
			#print(newList)	
			objects = newList		
			objects = self.getRectangles2(newList,10)

			if len(objects) > 0:
				#print('objects = {}'.format(objects))
				for r in objects:
					
					w,h = r[2],r[3]
					x1,y1 = r[0]-10,r[1]-10
					x2,y2 = r[0]+w+10,r[1]+h+10

					pts1 = np.float32([[x1,y1],[x1,y2],[x2,y2]])

					pts2 = np.float32([[0,0],[0,h+10],[w+10,h+10]])

					#print('pts1 ={}, pts2 ={}'.format(pts1,pts2))

					M = cv2.getAffineTransform(pts1,pts2)

					#blur = cv2.GaussianBlur(frame_gray,(-1,-1),2)
					#edges = cv2.Canny(blur,10,20)

					img3 = cv2.warpAffine(current_frame,M,(w+10,h+10))

					frame_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

					#cv2.imshow('Afine',img2)
					cv2.imshow('object',img3)

					#print('calling matching')
					#	print("imge = {}".format(img2))
					isMatch,hasKeyPoints = self.matching(frame_gray,matchMargin)
					if not isMatch :
						print('isMatched = {},self.objectList.len={}, len(objects) = {}'.format(isMatch,len(self.objectList),len(objects) ))
						finalObjectsLoc.append([x1+10,y1+10,x2-10,y2-10])
						finalObjects.append(img3)

						#print('final objects = {}'.format(finalObjects))

					#cv2.rectangle (current_frame,(tl[0],tl[1]),(br[0],br[1]),(0,0,255),1)
		#print(finalObjects)
		return finalObjectsLoc,finalObjects


	def checkMatches(self, matches):

		if len(matches) == 0:
			return False

		maxDist = max(x.distance for x in matches)
		minDist = min(x.distance for x in matches)
		count = sum(( x.distance <= minDist+0.50*maxDist) for x in matches)
		#print('count = {}, matches count = {}, maxDist = {}, minDist = {}'.format(count, len(matches),maxDist,minDist))
		if len(matches) >=1:
			return True
		
		
		if count>=80 *len(matches):
			return True,count,minDist,maxDist

		return False,count,minDist,maxDist
	def matching(self,ob, matchMargin):
	    count = 0
	    isMatched = False
	    hasKeyPoints = True
	    #orb = cv2.ORB_create()
	    if len(self.objectList) == 0:
	    	#print('ob={}'.format(ob))
	        self.objectList.append(ob)
	        #print('ob = {}'.format(ob))
	    else:
	        for listImg in self.objectList:

	        	 # Initiate STAR and BRIEF
	            star = cv2.FeatureDetector_create("SIFT")
	            brief = cv2.DescriptorExtractor_create("SIFT")

	            #print('calling star detect')

	            # find the keypoints and descriptors with SIFT
	            kp1 = star.detect(ob,None)

	            #print('detect 1 complete')
	            kp2 = star.detect(listImg,None)
	            #print('detect 2 complete')
	            #print('length of keypoints ={}'.format(len(kp1)))

	            if len(kp1) <=5:
	            	hasKeyPoints = False
	            	isMatched = True
	            	break
	            else:
	            	hasKeyPoints = True

	            #print('compute calling - len of kp ='+str(len(kp1)))

	            #print('problem with ob')
	            kp1, desc1 = brief.compute(ob, kp1)


	            kp2, desc2 = brief.compute(listImg,kp2)
	            #cv2.imshow('ob',ob)
	            #print('listImg ={}, ob ={}'.format(desc2,desc1))
	            # BFMatcher with default params
	            
	            if desc1 is not None and desc2 is not None:

	            	FLANN_INDEX_KDTREE = 0
	            	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	            	search_params = dict(checks=50)   # or pass empty dictionary
	            	flann = cv2.FlannBasedMatcher(index_params,search_params)
	            	matches = flann.knnMatch(desc1,desc2,k=2)
	            	'''bf = cv2.BFMatcher()
	            	matches = bf.knnMatch(desc1,desc2, k=2)'''
	            	if len(matches) > 0:
		            	# Apply ratio test
		            	good = []
		            	#print('matches = {}'.format(len(matches)))
		            	for m,n in matches:
		            		#print('m.distance ={}, n.distance ={}'.format(m.distance,n.distance))
		            		if m.distance <= 0.8*n.distance:
		            			good.append([m])
		            	'''if len(matches) <=20:
		            		margin = 7
		            	else:
		            		margin = int(0.15*len(matches))'''

		            	margin = matchMargin*len(matches)

		            	if len(good) >=margin:
		            		isMatched = True
		            		break
		        else:
		        	#not Descriptors means no interesting points
		        	isMatched = True
	            	
	        if not isMatched:
	        	print('good = {}, matches={},margin={}'.format(len(good), len(matches),margin))
	        	if hasKeyPoints:
	        		self.objectList.append(ob)

	    return isMatched,hasKeyPoints

if __name__ == '__main__':
 	mo = MotionObjects()
 	mask = cv2.imread('fristFrame.png',0)
 	image = cv2.imread('img2.png')
 	objects = mo.getObjects(mask,image)
 	print(objects)

    
        
    
    
    
