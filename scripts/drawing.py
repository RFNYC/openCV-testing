# importing open cv
import cv2 as cv

image = cv.imread('../assets/important-photo.jpg', 1)

image = cv.resize(image, (0, 0), fx=0.5, fy=0.5)
shapeinfo = image.shape

height = shapeinfo[0]
width = shapeinfo[1]

# Drawing a line. Args: image, starting-pixel, length of the line, color (BGR), thickness
# width 364, height 290 - this makes a diagonal line?? look into the docs for these methods
image = cv.line(image, (0, 0), (width, height), (255,0,0), 10)

# Same idea here. pass an image, give top right corner, then bottom right corner, and itll draw a cube around those coordinates. Passing -1 will fill the cube.
img = cv.rectangle(image, (100,100), (200, 200), (128,128,128), 3)
# circles take a CENTER first, and then it draws the line around your given radius.
img = cv.circle(img, (120, 120), 20, (0,0,255), 2)


# Drawing text:
font = cv.FONT_HERSHEY_SIMPLEX
# pass bottom lefthand corner pixel... ngl im not explaining this LOL, check the api/docs idk.
img = cv.putText(img, 'robert is great', (110, 40), font, 1, (0, 0, 0), 4, cv.LINE_AA)

cv.imshow("Display window", img)

print(image.shape)

k = cv.waitKey(0)

# closes the window
cv.destroyAllWindows()


