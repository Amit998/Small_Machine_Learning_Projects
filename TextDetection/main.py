import cv2
import pytesseract


pytesseract.pytesseract.tesseract_cmd='C:\Program Files (x86)\Tesseract-OCR\\tesseract.exe'

img=cv2.imread('img/1.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

print(pytesseract.image_to_string(img))
# var=(pytesseract.image_to_boxes(img))
# print(var)

# for i in var:
#     print(i,end='')
# print(len(var))

# ## Detecting Characters

imgh,imgw,_=img.shape

# boxes=(pytesseract.image_to_boxes(img))
# templist=[]

# for b in boxes.splitlines():

#     b=b.split(' ')
#     templist.append(b)
#     x,y,w,h=int(b[1]),int(b[2]),int(b[3]),int(b[4])
#     print(x,y,h,w)
#     cv2.rectangle(img,(x,imgh-y),(w+10,imgh-h),(0,0,255),3)
#     cv2.putText(img,b[0],(x,imgh-y+30),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,250),2)

    # cv2.rectangle(img,(x,y),(w,h),(0,0,255),1)

    # templist.append(b)
    # print(b)



# x,y,w,h=int(templist[0][1]),int(templist[0][2]),int(templist[0][3]),int(templist[0][4])


# cv2.rectangle(img,(x,imgh-y),(w,imgh-h),(0,0,255),1)
# print(templist[0][0])
# print(x,y,h,w)
# print(imgh,imgw)
# print(h+12)
# cv2.rectangle(img,(x,y),(w,h),(0,0,255),1)
# cv2.rectangle(img,(311,129),(380,250),(0,0,255),1)2

# print(templist[0][1],,)

### Detecting Words

cong=r'--oem 3 --psm 6 outputbase digits'
boxes=(pytesseract.image_to_data(img,config=cong))
# print(boxes)


for x,b in enumerate(boxes.splitlines()):
    # print(x,b)
    if(x !=0):
        # print(x)
        b=b.split()
        # print(b)
        if(len(b) == 12):
            print(b[6:])
            x,y,w,h=int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(img,(x,y),(w+x,h+y),(0,0,255),3)
            cv2.putText(img,b[11],(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,250),2)






cv2.imshow('Result',img)
cv2.waitKey(0) 