import cv2 as cv
import random

image = cv.imread('../assets/important-photo.jpg', 1)
image = cv.resize(image, (0, 0), fx=0.5, fy=0.5)

# For reference:
# print(image.shape)  =>  OUTPUT: (290, 364,  3)
#                                  ROW  COL  RGB

# This loop changes the colors of each individual pixels to be a random color. This creates a static-like effect

# for the first 100 rows
for i in range(100):
    # run this effect for x many columns in image.   
    for j in range(image.shape[1]):
        # at row I, choose column J and set pixel info = [R, G, B] -- random.randint(range) chooses a number from any range you set. like 0 -> 255
        image[i][j] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

cv.imshow("Display window", image)

k = cv.waitKey(0)

# closes the window
cv.destroyAllWindows()


