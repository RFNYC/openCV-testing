# importing open cv
import cv2 as cv

# OpenCV is going to be used to display images. For video you are essentially displaying X amount of images each for about a millisecond.
# This gives the effect of video / moving picture... I'll start by creating an image.

# initializing frame/image variable for display. 
# The second argument selects the color channel thingy (e.g. full color, grayscale, transparency)
image = cv.imread('../assets/important-photo.jpg', 1)

# (0, 0), fx=__ , fy=__  allows you to resized relative to the original size. 0.5 meaning half the original size.
image = cv.resize(image, (0, 0), fx=0.5, fy=0.5)

# opens a window. first arg: window name, second arg: display variable
cv.imshow("Display window", image)

print(image.shape)

# .waitKey() stops any code beyond this line from happening until a keystroke occurs OR the time limit was reached.
# a time limit of zero means theres infinite time. otherwise it will wait X seconds to pass this line if a keystroke isnt pressed.
k = cv.waitKey(0)

# closes the window
cv.destroyAllWindows()


