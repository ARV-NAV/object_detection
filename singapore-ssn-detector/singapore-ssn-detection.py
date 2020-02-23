# Starting code by Jean Vitor, available at https://jeanvitor.com/tensorflow-object-detecion-opencv/
import cv2
import numpy as np

# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow(
        'ssd_inception_v2_smd_2019_01_29/frozen_inference_graph.pb'
        , 'ssd_inception_v2_smd_2019_01_29/graph.pbtxt')

# Set up a list of class labels. There's a tensorflow method,
# but in this case I'm just creating a list since there's only
# 10 classes... Refactor later. See ./ssd_inc...1_29/labels.pbtxt
# Upon further experimentation, this list seems to be out of order
labels = ['Ferry',
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
colors = np.random.randint(0, 255, size=(len(labels), 3),
	dtype="uint8")

# This is the main loop that processes frames from video cameras
# or input video. For a camera, use:
# cap = cv2.VideoCapture(**camera number**)
cap = cv2.VideoCapture("MVI_0797_VIS_OB.avi")
while(True):
    # Input image
    ret, img = cap.read()
    if not ret:
        break

    h, w = img.shape[:2]
    #crop img to square for use in detector
    #img = img[0:h, int((w/2)-(h/2)):int((w/2)+(h/2))]
    #img = img[0:h, 0:int(h)]

    rows, cols, channels = img.shape

    # Use the given image as input, which needs to be blob(s).
    # Originally, parameters for blob were
    # blobFromImage(img,size=(300,300), swapRB=True, crop=True)
    # This seems to work better
    tensorflowNet.setInput(cv2.dnn.blobFromImage(cv2.resize(img,(300,300)), swapRB=True, crop=True))

    # Runs a forward pass to compute the net output
    networkOutput = tensorflowNet.forward()

    # Loop on the outputs
    for detection in networkOutput[0,0]:

        score = float(detection[2])
        if score > 0.5:
            objID = int(detection[1])+1
            print("Found a " + labels[objID] + " with confidence " + str(detection[2]))
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows

            #draw a coloured rectangle around object.
            # rectangle did not play nice with numpy array, hence manual casting
            theColor = (int(colors[objID][0]), int(colors[objID][1]), int(colors[objID][2]))
            # Draw some text too
            text = "{}: {:.4f}".format(labels[objID], detection[2])
            cv2.putText(img, text, (int(left), int(top) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, theColor, 2)
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), theColor, thickness=2)

    # Show the image with a rectagle surrounding the detected objects
    h, w = img.shape[:2]
    scaledImg = cv2.resize(img,(900, int((h/w)*900)))
    cv2.imshow('Image', scaledImg)

    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cv2.destroyAllWindows()
