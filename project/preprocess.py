import cv2
import os
import numpy as np
import csv as csv

data=csv.reader(open('0.csv','r'))
d=[]
for j in data:
	d.append([j[0],j[1]])
def smooth(img):
	dest=cv2.medianBlur(img,7)
	return dest

def process(path,img):
	image=cv2.imread(path+img,1)
	image=smooth(image)
	print image
	return image
	
def kmeans(img):
	output=[]
	image=img.reshape(img.shape[0]*img.shape[1],3)
	image=np.float32(image)
	nclusters=6
	criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
	attempts=10
	flags=cv2.KMEANS_RANDOM_CENTERS
	compactness,labels,centers=cv2.kmeans(image,nclusters,None,criteria,attempts,flags)
	print labels
	print centers
	print labels.flatten()
	centers = np.uint8(centers)
	res = centers[labels.flatten()]
	res2 = res.reshape((img.shape))
	params=cv2.SimpleBlobDetector_Params()
	params.filterByArea = True
	params.minArea=250
	params.maxArea=2500
	params.filterByConvexity = False	
	params.minConvexity=0.95	
	params.filterByCircularity = True
	params.minCircularity=0.65
	detector=cv2.SimpleBlobDetector_create(params)
	keypoints=detector.detect(res2)
	maxd=0
	for i in range (0,len(keypoints)):
		x=int(keypoints[i].pt[0])
		y=int(keypoints[i].pt[1])
		dm=int(keypoints[i].size)
		maxd=max(maxd,dm)
		print x,' ',y,' ',dm
		for j in range(0,len(d)):
			if(x>=float(d[j][0])-200 and x<=float(d[j][0])+200 and y>=float(d[j][1])-200 and y<=float(d[j][1])+200):
				output.append(res2[y-dm:y+dm,x-dm:x+dm])
				cv2.imwrite(dest+str(i)+'.png', cv2.resize(res2[y-dm:y+dm,x-dm:x+dm],(70,70)))
				d.pop(j)
				break
		
	im_with_keypoints=cv2.drawKeypoints(res2,keypoints,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
	'''
	lowerred=np.array([0,0,255])
	darkred=np.array([0,0,255])
	im2=cv2.inRange(im_with_keypoints,lowerred,darkred)
	ret,thresh = cv2.threshold(im2,127,255,0)
	_, contours, _= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	print len(contours)
	for i in range(0, len(contours)):
		cnt = contours[i]
		x,y,w,h = cv2.boundingRect(cnt)
		for j in range(0,len(d)):
			if(x>=float(d[j][0])-200 and x<=float(d[j][0])+200 and y>=float(d[j][1])-200 and y<=float(d[j][1])+200):
				cv2.imwrite(dest+str(i)+'.png', res2[y-25*h:y+25*h,x-25*w:x+25*w])
				d.pop(j)
				break
	'''	
	cv2.imshow('Keypoints',im_with_keypoints)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return output,maxd
	
def preprocess(path):
	images=[]
	count=0
	maxd=0
	print "Median Blur"
	for i in os.listdir(path):
		images.append(process(path,i))
		count+=1
	print "K Means"
	for i in range(0,count):
		images[i],m=kmeans(images[i])
		maxd=max(maxd,m)
	'''for i in range(0,count):
		for j in range(0,len(images[i])):
			w,h=images[i][j].size()
			w=maxd-int(w/maxd)
			h=maxd-int(h/maxd)
			cv2.imwrite(dest+str(i)+str(j)+'.png', cv2.resize(images[i][j],None,fx=w,fy=h,interpolation=cv2.INTER_LINEAR))'''
		

dest='../train/'
print "Preprocess"
preprocess('../preprocess/')
