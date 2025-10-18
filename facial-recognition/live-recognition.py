import face_recognition
import cv2 as cv
import numpy as np

# Initialize video variable & open webcam
capture = cv.VideoCapture(0)

# Loading known people & saving their face patterns
robert_img = face_recognition.load_image_file('../assets/known-people/robert.jpeg')
jay_img = face_recognition.load_image_file('../assets/known-people/jaynp.jpg')

robert_encoding = face_recognition.face_encodings(robert_img)
jay_encoding = face_recognition.face_encodings(jay_img)

# I think a some hashmap solution would be needed if scaled up.
known_faces = [
    robert_encoding,
    jay_encoding
]
known_names = [
    "Robert Feliciano",
    "Jay Noppone P."
]

# Initialize place to store face locations, patterns, & names for later use.
face_locations = []
face_encodings = []
face_names = []

# Suggested optimization: Switch this variable from true to false every frame. Means every *other* frame is processed
process_this_frame = True

while True:
    # begin reading webcam and return frames
    ret, frame = capture.read()

    # Every other frame logic: at the end of this loop we set this var != itself, so false.
    if process_this_frame == True:

        # Suggested optimization 2: Resize video to 1/4 for faster processing speed. Will be sized back up.
        small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Since openCV uses BGR by default it needs to be converted to RGB for facial-recognition.
        # note - i have no idea how this works :sob:
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find every face in the frame and then encode them (save pattern)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        # Encodes every face found as they're found
        face_encodings = face_recognition.face_locations(rgb_small_frame, face_locations)

        # Array for people currently present:
        face_names = []

        for face_pattern in face_encodings:
            # See if the face is a match for any of the known faces
            matches = face_recognition.compare_faces(known_faces, face_pattern)

            # Initializing failure var
            name = 'Unknown'


            # note - I don't have a strong handle on what this block does either so this is pasted.

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_faces, face_pattern)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

            face_names.append(name)

    # setting frame process = false so the next frame doesn't get processed.
    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
capture.release()
cv.destroyAllWindows()