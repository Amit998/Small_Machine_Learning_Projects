import cv2
import numpy as np

cap=cv2.VideoCapture(0)
whT=320
nmsTH=0.3
confidenceTH=0.5

classFile='coco.names'
classNames=[]

with open(classFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')
# print(classNames)
length=len(classNames)


modelConfiguration='yolov3-320.cfg'
modelWeights= 'yolov3.weights'

# modelConfiguration='yolov3-tiny.cfg'
# modelWeights= 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



def findObj(outputs,img):
    ht,wt,ct=img.shape
    bbox=[]
    classIds=[]
    confs=[]

    for output in outputs:
        for detection in output:
            scores=detection[5:]
            classId=np.argmax(scores)
            confidence=scores[classId]

            if(confidence > confidenceTH):
                w,h=int(detection[2] * wt),int(detection[3] *ht)
                x,y= int(int(detection[0] * wt) - w/2), int(int(detection[1] * ht) - h/2)
                print([x,y,w,h])
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    # print(len(bbox))
    # print(confs)
    # print(classIds)
    indices=cv2.dnn.NMSBoxes(bbox,confs,confidenceTH,nmsTH)
    print(indices)

    for i in indices:
        i = i[0]
        box=bbox[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        print(x,y,w,h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {round(float(confs[i]) *100,2) } %',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
        # print(f'{classNames[classIds].upper} {float(confs[i]) *100 } ','%')
        # print(f'{classNames[classIds[i]].upper}')


while True:
    success,img=cap.read()

    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerName=net.getLayerNames()
    # print(layerName)
    # print(net.getUnconnectedOutLayers())
    outPutNames=[layerName[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(outPutNames)

    outPuts=net.forward(outPutNames)
    findObj(outPuts,img)
    # print(outPuts[0][0])
    # print(outPuts[1].shape)
    # print(outPuts[2].shape)

    cv2.imshow('Image',img)
    cv2.waitKey(1)