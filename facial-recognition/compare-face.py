from PIL import Image
import face_recognition

# initialize photo & find faces
robert = face_recognition.load_image_file('../assets/known-people/robert.jpeg')
unknown_image = face_recognition.load_image_file('../assets/unknown/redbull.jpeg')

# .face_encodings(x) returns an array of these sort of digital finger prints for each face it finds in the image.
# you can compare these digital finger prints to tell if you found a match for a specific face.
try:
    robert_face_encoding = face_recognition.face_encodings(robert)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except:
    print("No faces were found or some other error occured.")

print(robert_face_encoding)
print(unknown_face_encoding)

results = face_recognition.compare_faces([robert_face_encoding], unknown_face_encoding)

if results[0] == True:
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")