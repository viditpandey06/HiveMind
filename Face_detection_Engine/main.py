import cv2
def faceBox(faceNet, ageNet, genderNet, frame):
    print("FaceBox")
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bbox = []
    confidenceThreshold = 1.0  # Adjust this value to increase or decrease the confidence threshold
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > confidenceThreshold:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameWidth)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameWidth)
            face = frame[y1:y2, x1:x2]
            bbox.append([x1, y1, x2, y2])
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            cv2.putText(frame, gender, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameWidth)
            face = frame[y1:y2, x1:x2]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [104, 117, 123], swapRB=False)
            ageNet.setInput(blob)
            genderNet.setInput(blob)
            agePred = ageNet.forward()
            genderPred = genderNet.forward()
            age = agePred[0][0][0][0] * 100
            gender = "Male" if genderPred[0][0][0][0] < 0.5 else "Female"
            bbox.append([x1, y1, x2, y2, age, gender])
    return bbox

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
print("FaceNet")
ageNet = cv2.dnn.readNet(ageModel, ageProto)
print("AgeNet")
genderNet = cv2.dnn.readNet(genderModel, genderProto)
print("GenderNet")
video = cv2.VideoCapture(0)
while True:
    print("While")
    ret, frame = video.read()
    bbox = faceBox(faceNet, ageNet, genderNet, frame)
    for (x1, y1, x2, y2, age, gender) in bbox:
        print("For")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Age: {int(age)} Gender: {gender}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Age_Gender", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()