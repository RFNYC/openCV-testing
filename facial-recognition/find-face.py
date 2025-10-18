from PIL import Image
import face_recognition

# Finding the individual faces in an image:
image = face_recognition.load_image_file('../assets/unknown/redbull.jpeg')

# returns an array of tuples. each tuple has 4 numbers which contain bounding information for each face. 
# the numbers will correlate to the furthest row & column the face reaches, from there you can find the coordinate points of each corner by
# combining the furthest points. 
face_locations = face_recognition.face_locations(image)

# typical output: [(719, 867, 793, 793)] | Order values: [(Top, Right, Bottom, Left)] 
# Left & Right act as X (column) values, Top and Bottom act as Y (row) values. Thus:
# Bottom left => (Left, Bottom) if we're passing it to openCV...I hope?
print(face_locations) 

# For every set of bounding information found...
for bounding_set in face_locations:

    # from the bounding set assign these variables.  reminder: array looks like this [(0, 1, 2, 3)] and variables are assigned in this order.
    top, right, bottom, left = bounding_set

    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()