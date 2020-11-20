import cv2
import numpy as np
import face_recognition

imgBill=face_recognition.load_image_file('img/Bill Gates.jpg')
imgBill=cv2.cvtColor(imgBill,cv2.COLOR_BGR2RGB)


imgBill2=face_recognition.load_image_file('img/Elon Mask.jpg')
imgBill2=cv2.cvtColor(imgBill2,cv2.COLOR_BGR2RGB)


faceLoc=face_recognition.face_locations(imgBill)[0]
endcodeBill=face_recognition.face_encodings(imgBill)[0]
cv2.rectangle(imgBill,(faceLoc[3], faceLoc[0]),(faceLoc[1], faceLoc[2]),(255,0,255),2)
# print(faceLoc)


faceLocTest=face_recognition.face_locations(imgBill2)[0]
endcodeBillTest=face_recognition.face_encodings(imgBill2)[0]
cv2.rectangle(imgBill2,(faceLocTest[3], faceLocTest[0]),(faceLocTest[1], faceLocTest[2]),(255,0,255),2)


result=face_recognition.compare_faces([endcodeBill],endcodeBillTest)
face_dis=face_recognition.face_distance([endcodeBill],endcodeBillTest)
print(face_dis)

cv2.putText(imgBill2,f'{result} {round(face_dis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
# cv2.imshow('Bill Gates',imgBill)
cv2.imshow('Bill Gates Test',imgBill2)
cv2.waitKey(0)