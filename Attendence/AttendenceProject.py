from os import sep
import cv2
from face_recognition.api import face_locations
import numpy as np
import face_recognition
import os
from  datetime import datetime


path ='img'
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)
# print(path)

for cls in myList:
    currImage=cv2.imread(f'{path}/{cls}')
    # print(currImage)

    images.append(currImage)
    classNames.append(os.path.splitext(cls)[0])




# print(classNames)


def findEncoding(images):
    encodList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodList.append(encode)

    return encodList


encodeListKnown=findEncoding(images)
print('Encoding Complete')
# print(encodeListKnown)


def mark_Attendence(name):
    with  open('attendence.csv','r+') as f:
        nameList=[]

        myDataList=f.readlines()
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            
            dtString=now.strftime("%m/%d/%Y, %H:%M:%S")
            f.writelines(f'\n{name},present,{dtString}')

        # print(myDataList)

# mark_Attendence('AMIT DUTTA')

cap=cv2.VideoCapture(0)

while True:
#     print('In First Loop')
    success,img =cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    
   
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    # faceCurFrame=face_recognition.face_locations(imgS)
    faceLoc=face_recognition.face_locations(imgS)
    encodeCurFace=face_recognition.face_encodings(imgS,faceLoc)
    # faceLoc=face_recognition.face_locations(imgS)
#     print('In Second')

    for encodeFace,faceLoc in zip(encodeCurFace,faceLoc):
        print('In Third')
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDistance=face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDistance)
        matchIndex=np.argmin(faceDistance)

        if matches[matchIndex]:
            # print('In 4th')
            name=classNames[matchIndex].upper()
            y1,y2,x1,x2=faceLoc
            y1,y2,x1,x2=y1*4,y2*4,x1*4,x2*4

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y1-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255),2)

            mark_Attendence(name)

            
            print(name)
    cv2.imshow('Webcame',img)
    

    cv2.waitKey(1)
    
    




# faceLoc=face_recognition.face_locations(imgBill)[0]
# endcodeBill=face_recognition.face_encodings(imgBill)[0]
# cv2.rectangle(imgBill,(faceLoc[3], faceLoc[0]),(faceLoc[1], faceLoc[2]),(255,0,255),2)
# # print(faceLoc)


# faceLocTest=face_recognition.face_locations(imgBill2)[0]
# endcodeBillTest=face_recognition.face_encodings(imgBill2)[0]
# cv2.rectangle(imgBill2,(faceLocTest[3], faceLocTest[0]),(faceLocTest[1], faceLocTest[2]),(255,0,255),2)


# result=face_recognition.compare_faces([endcodeBill],endcodeBillTest)
# face_dis=face_recognition.face_distance([endcodeBill],endcodeBillTest)

# imgBill=face_recognition.load_image_file('img/Bill Gates.jpg')
# imgBill=cv2.cvtColor(imgBill,cv2.COLOR_BGR2RGB)


# imgBill2=face_recognition.load_image_file('img/Bill-Gates-2011.jpg')
# imgBill2=cv2.cvtColor(imgBill2,cv2.COLOR_BGR2RGB)