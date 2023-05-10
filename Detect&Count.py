import cv2
import torch
import numpy as np
import time 

points=[]
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)            
    
           


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture("videos/test1.mp4")
count=0


area = [(0,640),(906,636),(610,100),(420,90)]

while True:
    ret,frame=cap.read()
    if not ret:
        break
    #frame=cv2.resize(frame,(600,1200))
    frame = frame[800: ,:]

    results=model(frame)
    Cars = []
    Trucks = []
    Buses = []
    Motorcycles = []
    Bicycles = []
    Persons = []
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d=(row['name'])
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        if 'car' in d:
           results = cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
           if results>=0:
              cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
              cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),1)
              Cars.append([cx])
           
        elif 'truck' in d:
            results = cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
            if results>=0:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),1)
                Trucks.append([cx])
           
        elif 'bus' in d:
            results = cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
            if results>=0:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),1)  
                Buses.append([cx])
           
        elif 'motorcycle' in d:
            results = cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
            if results>=0:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),1)   
                Motorcycles.append([cx])  
           
        elif 'bicycle' in d:
            results = cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
            if results>=0:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),1) 
                Bicycles.append([cx]) 
           
        elif 'person' in d:
            results = cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
            if results>=0:
                 cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                 cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),1) 
                 Persons.append([cx])
                 
    cv2.polylines(frame,[np.array(area,np.int32)],True,(0,255,0),2)
    a = len(Cars)
    b = len(Trucks)
    c = len(Buses)
    d = len(Motorcycles)
    e = len(Bicycles)
    f = len(Persons)
    #print("Cars : {}".format(len(Cars)))
    #print("Trucks : {}".format(len(Trucks)))
    #print("Buses : {}".format(len(Buses)))
    # print("Motorcycles : {}".format(len(Motorcycles)))
    # print("Bicycles : {}".format(len(Bicycles)))
    # print("person : {}".format(len(Persons)))
   
    total = a + b+c +d+e
    cv2.putText(frame,str("Cars : {}".format(len(Cars))),(8,30),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    cv2.putText(frame,str("Trucks : {}".format(len(Trucks))),(8,60),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    cv2.putText(frame,str("Buses : {}".format(len(Buses))),(8,90),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    cv2.putText(frame,str("Motorcycles : {}".format(len(Motorcycles))),(8,120),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    cv2.putText(frame,str("Bicycles : {}".format(len(Bicycles))),(8,150),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    cv2.putText(frame,str("Total Vehicles: {}".format(total)),(8,180),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    #cv2.putText(frame,str(total),(255,30),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    cv2.imshow("FRAME",frame)
    cv2.setMouseCallback("FRAME",POINTS)
   
    #time.sleep(0)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

