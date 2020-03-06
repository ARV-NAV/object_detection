import numpy as np
import cv2 as cv

import sys
import os.path

camera1 = int(sys.argv[1])
camera2 = int(sys.argv[2])

cap1 = cv.VideoCapture(camera1)
cap2 = cv.VideoCapture(camera2)

fourcc = cv.VideoWriter_fourcc(*'XVID')
fourcc1 = cv.VideoWriter_fourcc(*'XVID')

filename1 = "output1"
filename2 = "output2"
while (os.path.exists(filename1 + ".avi") or os.path.exists(filename2 + ".avi")):
        filename1 = filename1 + "_"
        filename2 = filename2 + "_"
filename1 += ".avi"
filename2 += ".avi"

out1 = cv.VideoWriter(filename1,fourcc, 20.0, (640, 480))
out2 = cv.VideoWriter(filename2,fourcc, 20.0, (640, 480))

print("Press \'q\' to stop recording!")
while True:
    # Use grab and retrieve to have cameras closer to being synced timewise
    cap1.grab()
    cap2.grab()
    ret1, frame1 = cap1.retrieve()
    ret2, frame2 = cap2.retrieve()
    if ret1 == True:
        out1.write(frame1)
        out2.write(frame2)

        both = np.hstack((frame1,frame2))
        cv.imshow('video_capture', both)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap1.release()
cap2.release()
out1.release()
out2.release()
cv.destroyAllWindows()
