import numpy as np
import cv2 as cv

# initializing video variable & starting video
cap = cv.VideoCapture(0)

while True:
    # returns a numpy array representing each frame of video. 
    # refer to notes for numpy array explanation.

    # ret just lets you know if the camera is being used properly.
    ret, frame = cap.read()
    
    cv.imshow('Display window', frame)

    # waits 10 millisecond for each frame
    # if the keystroke returns a number that is equal to the ord value of q, the frame cycling loop will end.
    # essentially meaning if you pressed q do this thing.

    # cv.waitKey(x) monitors keystrokes and it delays the next openCV frame by X milliseonds. So if you want to save processing power
    if cv.waitKey(10) == ord('q'):
        break

# closing video
cap.release()
# close display window
cv.destroyAllWindows()