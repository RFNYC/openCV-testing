# importing open cv
import cv2 as cv

image = cv.imread('../assets/important-photo.jpg', 1)

image = cv.resize(image, (0, 0), fx=0.5, fy=0.5)
shapeinfo = image.shape

height = shapeinfo[0]
width = shapeinfo[1]

# Using the picture of the dog again. In any case where we're drawing a square/rectangle around an object we're detecting we'll have a top left
# and bottom right coordinate. They will be in the form of (x1, y1) & (x2, y2) => (column1, row1) & (column2, row2)

# For this template we'll use hardcoded values since I'm only worried about making the square look good, not its position.
topLeft = (80,30)
bottomRight = (220,180)

# Drawing the border around the dog
img = cv.rectangle(image, topLeft, bottomRight, (0,0,255), 2)

# Drawing the caption box below the border | We want the top left val to be the bottom left of the border.
capHeight = 20 # height of caption in pixels

# +- 1 is padding - feel free to change if you forgot how to use it
capTL = (topLeft[0]-1, bottomRight[1])  # top-left coordinate of the captionbox
capBR = (bottomRight[0]+1, bottomRight[1]+capHeight)  # bottom-left coordinate of the captionbox
img = cv.rectangle(img, capTL, capBR, (0,0,255), -1)

# Drawing text:
font = cv.FONT_HERSHEY_SIMPLEX
# pass bottom lefthand corner pixel and the height of the font. always include cv.LINE_AA
textBL = (topLeft[0], bottomRight[1]+15)
img = cv.putText(img, 'important dog', textBL, font, .5, (0, 0, 0), 1, cv.LINE_AA)


cv.imshow("Display window", img)

cv.waitKey(0)
cv.destroyAllWindows()

