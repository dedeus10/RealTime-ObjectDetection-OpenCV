import cv2

img = cv2.imread('lena.png')

classNames = []
class_file = 'coco.names'

with open(class_file,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weights, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, conf, bbox = net.detect(img, confThreshold=0.6)
for classId, confidence, box in zip(classIds.flatten(), conf.flatten(), bbox):
    cv2.rectangle(img, box, color=(0,255,0), thickness= 3)
    cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX,
                2, (0,255,0), 2)

cv2.imshow("Output",img)
cv2.waitKey(0)