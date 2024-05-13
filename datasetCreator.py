import cv2
import numpy as np


faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);

id = input('Enter User ID: ')   #id variable

sampleNum=0
Max_sample = 100

while(True):
	ret,img=cam.read();
	#print(img)
	#print(ret)
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	cv2.imshow("gray",gray);
	
	faces=faceDetect.detectMultiScale(gray,1.3,5);
	for(x,y,w,h) in faces:
		sampleNum=sampleNum+1;
		cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])#whoever faces detect it will write that face for that use a id variable
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		cv2.waitKey(100);
	cv2.imshow("Face",img);
	cv2.waitKey(1);
	if(sampleNum>Max_sample):
		break
cam.release()
cv2.destroyAllWindows()
