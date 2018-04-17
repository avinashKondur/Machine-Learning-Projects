'''
MOSSE tracking sample
This sample implements correlation-based tracking approach as described in [1].
Usage:
  mosse.py [--pause] [<video source>]
 
[1] David S. Bolme et al. "Visual Object Tracking using Adaptive Correlation Filters"
    http://www.cs.colostate.edu/~bolme/publications/Bolme2010Tracking.pdf
'''

# Python 2/3 compatibility
import  MotionObjects 

import imp
imp.reload(MotionObjects)

from MotionObjects import MotionObjects
from StillObjects import StillObjects
from Objects import Objects

from CSVHelper import writeOutput,CreateFile

from vgg16 import vgg16
import tensorflow as tf
from imagenet_classes import class_names

from scipy.misc import imread, imresize
from collections import Counter

import sys

import numpy as np
import math

import cv2
#from common import writeText, RectSelector
#import video

def Apply_affine(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv2.warpAffine(a, T, (w, h), borderMode = cv2.BORDER_REFLECT)

def CalculateFilter(A, B):
    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C

def writeText(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))

eps = 1e-5

class MOSSE:
    def __init__(self, frame, rect,trackNo, frameNo):
        x1, y1, x2, y2 = rect

        self.startpos = (x1,y1)
        self.startFrame = frameNo

        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        #self.lastPos = self.pos
        self.size = w, h
        self.objectMoved = False
        self.objectLeft = False
        self.stationaryCount = 0
        #self.isOverLap = False
        #print(frame,w,h,x,y)

        self.trackNo = trackNo
        img = cv2.getRectSubPix(frame, (w, h), (x, y))
        self.obj=None

        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
       # print('in constructor - img = {}, img.shape = {}'.format(img, img.shape))
        for i in range(128):
            a = self.preprocess(Apply_affine(img))
            A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
            self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2 += cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.update_Filter()
        self.update(frame)

   

    def update(self, frame, rate = 0.125):
        (x, y), (w, h) = self.pos, self.size

        self.lastPos = self.pos
        #self.lastsize = self.size
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
        img = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)

        if dx!=0 or dy!=0:
            self.objectMoved = True

        
        self.good = self.psr >= 3.0
        if not self.good:
            return 

        self.pos = x+dx, y+dy

        self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
        img = self.preprocess(img)

        A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
        H2 = cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.H1 = self.H1 * (1.0-rate) + H1 * rate
        self.H2 = self.H2 * (1.0-rate) + H2 * rate
        self.update_Filter()

    def draw_track(self, frame):
        #print('self.objectMoved= {}'.format(self.objectMoved))
        #print('isObjectLeft={}'.format(self.objectLeft))
        if not self.objectLeft :
           # print('object @ {}()'.format(self.pos,self.size))
            (x, y), (w, h) = self.pos, self.size
            x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)

            #if self.trackNo == 1:
                #print('x1={},y1={}, x2 ={},y={}, vis.shape[0]={}'.format(x1,y1,x2,y2,vis.shape))
            self.checkIfObjectLeft(frame)
            #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pts1 = np.float32([[x1,y1],[x1,y2],[x2,y2]])

            pts2 = np.float32([[0,0],[0,224],[224,224]])

            #print('pts1 ={}, pts2 ={}'.format(pts1,pts2))

            M = cv2.getAffineTransform(pts1,pts2)
            self.obj = cv2.warpAffine(frame,M,(224,224))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))
            if self.good:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            else:
                self.stationaryCount +=1
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255))
                cv2.line(frame, (x2, y1), (x1, y2), (0, 0, 255))
            writeText(frame, (x1, y1), 'Track#: ' +str(self.trackNo))
        elif self.pos == self.lastPos:
            self.stationaryCount +=1

        if self.stationaryCount >=15:
            self.objectLeft = True

    def checkIfObjectLeft(self,frame):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        r,_,c = frame.shape
        #if object is on upper part of the screen
        if y1 < 0 or y2< 0:
            self.objectLeft = True
            return

        #if object is lower part of the screen
        if y1 >r+10 or y2 >r+10:
            self.objectLeft = True
            return

        #if object is left part of the screen

        '''if x1< -10 or x2<-10:
            self.objectLeft = True
            return
        # if object is right part of the screen
        if x1>w +10 or x2>w+10:
            self.objectLeft = True
            return'''
        return


    def get_Corners(self):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        if x1<0:
            x1=0
        if x2<0:
            x2=0
        if y1<0:
            y1=0
        if y2<0:
            y2=0
        return x1,y1,x2,y2

    def preprocess(self, img):
        #print('before preprocess img = {} - shape = {}'.format(img,img.shape))
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)

        #print('img = {} - shape  ={}, self.win = {} - shape = {}'.format(img,img.shape,self.win,self.win.shape))
        return img*self.win

    def correlate(self, img):
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)
        return resp, (mx-w//2, my-h//2), psr

    def update_Filter(self):
        self.H = CalculateFilter(self.H1, self.H2)
        self.H[...,1] *= -1

    def isGood(self):
        return self.good

    def isObjectLeft(self):
        return self.objectLeft

    def setOverLapped(self, bool):
        self.isOverLap = True

    def isOverlapped(self):
        return self.isOverLap


class ObjectTracking:

    def distance(self,p1, p2):
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        dist = math.hypot(x2 - x1, y2 - y1)
        return dist

    def __init__(self, input_video, output_video, output_file=None):
        self.cap = cv2.VideoCapture(input_video)

        self.output_video = output_video
        self.TrackingCSV = output_file

        CreateFile(self.TrackingCSV)
        
        _, self.frame = self.cap.read()

        self.trackers = []

        self.stillObjects = []
        self.motionObjects=[]

        self.Mimgs = []



    def getCenter(self,tl,br):
        x1, y1, x2, y2 = tl[0],tl[1],br[0],br[1]

        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])

        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
        x, y = x1+0.5*(w-1), y1+0.5*(h-1)

        return x,y

    def getWidth(self,tl,br):
        w, h = map(cv2.getOptimalDFTSize, [br[0]-tl[0], br[1]-tl[1]])

        return (w,h)

    def run(self):
        mo = MotionObjects()
        so = StillObjects()
        fgbg = cv2.BackgroundSubtractorMOG()
        
        ret, self.frame = self.cap.read()
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        
        blur = cv2.GaussianBlur(frame_gray,(-1,-1),2)
        fgmask = fgbg.apply(blur)
        fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
        out = cv2.VideoWriter(self.output_video,fourcc, 30, (self.frame.shape[1],self.frame.shape[0]), True)
        
        trackNo = 1
        framNo = 1

        identifiedObjs = []
        while True:
            print('trackers count = {}'.format(len(self.trackers)))
        
            
            if not ret:
                break

            #copy to last frame_gray
            self.last_frame = self.frame

            frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(frame_gray,(-1,-1),2)

            #edges = cv2.Canny(blur,10,30)

            fgmask = fgbg.apply(blur)

            #edges = cv2.Canny(fgmask,100,200)

            objectsLoc,objs = mo.getObjects(fgmask,self.frame,100,9999,5,0.15)


            self.motionObjects.extend(objs)

            if len(objectsLoc) > 0:
                for i in range(len(objectsLoc)):
                    #if len(self.trackers) ==0:
                    tracker = MOSSE(frame_gray, objectsLoc[i],trackNo,framNo)
                    obj = imresize(objs[i], (224, 224))
                    #print('obj.shape =', obj.shape)
                    identifiedObjs.append(Objects(obj,objectsLoc[i][0],objectsLoc[i][1],'MOVING',framNo,trackNo))
                    self.trackers.append(tracker)
                    trackNo+=1

            '''for traker in self.trackers:
                x1,y1,x2,y2 = tracker.get_Corners()
                writeOutput(self.TrackingCSV ,tracker.trackNo, framNo,x1,y1,x2,y2)'''

            #stillOb,tl_X,tl_y,br_X,br_y,kp = so.getStillObject(self.frame)
            #print(tl_X,tl_y,br_X,br_y)

            isSameObject = False
            for tracker in self.trackers:
                tracker.update(frame_gray)
                

            frame_copy = self.frame.copy()

            for tracker in self.trackers:
                if tracker.objectLeft:
                    self.trackers.remove(tracker)

            for tracker in self.trackers:
                tracker.draw_track(frame_copy)
                dist = self.distance(tracker.pos, (kp.pt[0],kp.pt[1]))

                if not isSameObject and  dist <= kp.size//2+ tracker.size[0]:
                    isSameObject = True
                x1,y1,x2,y2 = tracker.get_Corners()
                #print('tracker.obj.shape = ', tracker.obj.shape)
                identifiedObjs.append(Objects(tracker.obj,x1,y1,'MOVING',framNo,tracker.trackNo))

            if not isSameObject:
                identifiedObjs.append(Objects(stillOb,tl_X,tl_y,'STILL',framNo,None))
                cv2.rectangle(frame_copy,(tl_X,tl_y),(br_X,br_y),(0,255,0),3)

            #cv2.imshow('tracking frame', frame_copy)
            #cv2.imwrite('frame'+str(framNo)+'.png', frame_copy)
            out.write(frame_copy)

            ret, self.frame = self.cap.read()
            framNo+=1
        self.cap.release()
        cv2.destroyAllWindows()

        stillImgs = filter(lambda x:x.getObjectType()=='STILL',identifiedObjs)

        imgs = []
        for x in stillImgs:
            imgs.append(x.getObject())

        sess = tf.Session()
        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = vgg16(images, 'vgg16_weights.npz', sess)

        print('--------- Starting Still objects ------------------')
        image_stack = np.stack(imgs)

        probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack})

        preds = np.argmax(probs, axis=1)

        for index, p in enumerate(preds):
            stillImgs[index].setObjectLabel(class_names[p])
            stillImgs[index].setActivationLevel(probs[index, p])
            #print "Prediction: %s; Probability: %f"%(class_names[p], probs[index, p])

        sortList = sorted(stillImgs, key = lambda i:i.getActivationLevel(), reverse=True)

        selectedLabels = []
        top5 = []
        count = 0
        for o in sortList:
            if not o.objectLabel in selectedLabels:
                movtno = filter(lambda x:x.objectLabel == o.objectLabel, sortList)
                top5.append(movtno[0])
                selectedLabels.append(movtno[0].objectLabel)
                count+=1
                if count ==5:
                    break

        for t in top5:
            print t.objectLabel + str(t.getActivationLevel())
            writeOutput(self.TrackingCSV,t.startFrame,t.EndFrame,t.x,t.y, t.objectLabel,t.getActivationLevel())

        print('------------ End Still objects-------------')

        print('---------- Start Moving objects---------')


        movingObjs = filter(lambda x:x.getObjectType()=='MOVING',identifiedObjs)

        #Mimgs = []
        for x in movingObjs:
            print(x.getObject().shape)
            self.Mimgs.append(x.getObject())

        print(len(self.Mimgs))
        image_stack = np.stack(self.Mimgs)

        probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack})

        preds = np.argmax(probs, axis=1)

        for index, p in enumerate(preds):
            movingObjs[index].setObjectLabel(class_names[p])
            movingObjs[index].setActivationLevel(probs[index, p])

        finalMotionObjs =[]
        for tno in range(1,trackNo):
            movtno = filter(lambda x:x.trackNo == tno, movingObjs)
            endFrame = max(x.startFrame for x in movtno)
            movo = Objects(movtno[0].getObject(),movtno[0].x,movtno[0].y,'MOVING',movtno[0].startFrame,movtno[0].trackNo)
            movo.EndFrame = endFrame
            labelCount = Counter(x.objectLabel for x in movtno)
            maxLabel = max(labelCount)
            movo.setObjectLabel(maxLabel[0])
            movo.setActivationLevel(maxLabel[1]/len(movtno))
            finalMotionObjs.append(movo)

        for t in finalMotionObjs:
            print t.objectLabel + str(t.getActivationLevel())
            writeOutput(self.TrackingCSV,t.startFrame,t.EndFrame,t.x,t.y, t.objectLabel,t.getActivationLevel())
        print('------------ End Moving objects------------------')




if __name__ == '__main__':

    import sys
    args = sys.argv


    
    #indicates that arguments have been passed
    viedo_input  = args[1]
    video_output = args[2]

    output_file = args[3]



tracking = ObjectTracking(viedo_input,video_output,output_file).run()