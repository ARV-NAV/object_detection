# Starting code by Jean Vitor, available at https://jeanvitor.com/tensorflow-object-detecion-opencv/
import cv2
import numpy as np
from centroidtracker import CentroidTracker
from objData import objData
import datetime

# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow(
        'ssd_inception_v2_smd_2019_01_29/frozen_inference_graph.pb'
        , 'ssd_inception_v2_smd_2019_01_29/graph.pbtxt')

# Set up a list of class labels. There's a tensorflow method,
# but in this case I'm just creating a list since there's only
# 10 classes... Refactor later. See ./ssd_inc...1_29/labels.pbtxt
# Upon further experimentation, this list seems to be out of order
LABELS = ['Ferry',
          'Buoy',
          'Vessel/ship',
          'Speed boat',
          'Boat',
          'Kayak',
          'Sail boat',
          'Swimming person',
          'Flying bird/plane',
          'Other',
          '????',
          'dock?']

# initialize a list of colors to represent each possible class label
np.random.seed(38)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# Initialize centroid tracker. Needed to match objects between frames.
ct = CentroidTracker()

# This is the main loop that processes frames from video cameras
# or input video. For a camera, use:
# cap = cv2.VideoCapture(**camera number**)

# To test, I recomend using a video from the Singapore Maritime dataset.
# Files are too large for github, so must be downloaded independently.
#cap = cv2.VideoCapture("testvid.mp4")
cap = cv2.VideoCapture("MVI_0797_VIS_OB.avi")
while(True):
    # Input image
    ret, img = cap.read()
    if not ret:
        break

    h, w = img.shape[:2]
    # Scale image to reduce size for display later
    img = cv2.resize(img,(1200, int((h/w)*1200)))
    #crop img to square for use in detector
    #img = img[0:h, int((w/2)-(h/2)):int((w/2)+(h/2))]
    #img = img[0:h, 0:int(h)]

    rows, cols, channels = img.shape

    # Use the given image as input, which needs to be blob(s).
    # Originally, parameters for blob were
    # blobFromImage(img,size=(300,300), swapRB=True, crop=True)
    # This seems to work better
    tensorflowNet.setInput(
        cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),
        swapRB=True, crop=True))

    # Runs a forward pass to compute the net output
    networkOutput = tensorflowNet.forward()

    # Loop on the outputs
    rects = []
    for detection in networkOutput[0,0]:
        score = float(detection[2])
        if score > 0.5:
            # Subtract 1 since LABEL list is 0 indexed while DNN output is 1 indexed
            objID = int(detection[1])-1
            # print("Found a " + LABELS[objID] + " with confidence " + str(detection[2]))
            left = int(detection[3] * cols)
            top = int(detection[4] * rows)
            right = int(detection[5] * cols)
            bottom = int(detection[6] * rows)
            data = objData((left, bottom, right, top),
                           datetime.datetime.now().strftime("%H:%M:%S.%f"),
                           LABELS[objID],
                           detection[2],
                           int(COLORS[objID][0]))
            # centroid tracker takes format smallerX, smallerY, largerX, largerY
            # and a data object/tuple/structure/etc.
            rects.append([left, bottom, right, top, data])

    # Now, update the centroid tracker with the newly found bounding boxes
    (objects, data) = ct.update(rects)

    # Draw the objects being tracked
    ct.drawObjects(img)

    # Show the image with a rectagle surrounding the detected objects
    cv2.imshow('Image', img)

    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cv2.destroyAllWindows()
