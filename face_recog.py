## Load the training set

import numpy as np 
import cv2
import os

def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,query_x,k=5):
    vals=[]
    m=X.shape[0]
    
    for i in range(m):
        d=dist(query_x,X[i])
        vals.append((d,Y[i]))
        
    vals=sorted(vals)
    
    vals=vals[:k]
    
    vals=np.array(vals)
    new_vals=np.unique(vals[:,1],return_counts=True)
    index=new_vals[1].argmax()
    pred=new_vals[0][index]
    
    return pred



cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path='./data/'

face_data=[]
labels=[]

class_id=0

names={}

##Data Prepration

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):

        names[class_id]=fx[:-4]
        
        data_item=np.load(dataset_path+fx)
        print(type(data_item))
        print(data_item.shape)

        face_data.append(data_item)
       
        target = class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
print(type(face_dataset))
print(face_dataset.shape)
face_labels=np.concatenate(labels,axis=0)

while True:
    ret,frame = cap.read()
    if ret==False:
        continue

    faces = face_cascade.detectMultiScale(frame,1.3,5)
    for face in faces:
        x,y,w,h=face

        offset=10
        face_section = frame[y-offset:y+offset+h,x-offset:x+offset+w]
        face_section=cv2.resize(face_section,(100,100))


        out=knn(face_dataset,face_labels,face_section.flatten())
        
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-offset),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("F",frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
