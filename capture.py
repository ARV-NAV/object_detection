import numpy as np
import cv2 as cv

import sys
import os.path

#camera1 = int(sys.argv[1])

# Hacky workaround to make sure we get SOME video... Hopefully it's the right video!
camera1 = 1
cap1 = cv.VideoCapture(camera1, cv.CAP_DSHOW)
if cap1 is None or not cap1.isOpened():
    print("camera 1 not detected! defaulting to camera 0")
    camera1 = 0

cap1 = cv.VideoCapture(camera1, cv.CAP_DSHOW)

fourcc = cv.VideoWriter_fourcc(*'XVID')

filename1 = "output1"
while (os.path.exists(filename1 + ".avi")):
        filename1 = filename1 + "_"
filename1 += ".avi"

out1 = None

print("Press \'q\' to stop recording!")
while True:
    # Use grab and retrieve to have cameras closer to being synced timewise
    #cap1.grab()
    #ret1, frame1 = cap1.retrieve()
    ret1, frame1 = cap1.read()
    if out1 == None:
        (w, h) = frame1.shape[:2]
        print("Recording video at resolution " + str(w) + "x" + str(h))
        out1 = cv.VideoWriter(filename1,fourcc, 20.0, (h, w))
		
	
    if ret1 == True:
        out1.write(frame1)
        cv.imshow('video_capture', frame1)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap1.release()
out1.release()
cv.destroyAllWindows()
