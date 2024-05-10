import cv2
def faceBox(faceNet,frame):
    print(frame)
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame,1.0,(227,227),[104,117,123],swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bbox=[]
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameWidth)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameWidth)
            bbox.append([x1,y1])
    return detection

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"


faceNet=cv2.dnn.readNet(faceModel,faceProto)



video = cv2.VideoCapture(0)
while True:
    ret,frame=video.read()
    detect=faceBox(faceNet,frame)
    cv2.imshow("Age_Gender",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()