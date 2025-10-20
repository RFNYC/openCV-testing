import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import cv2 as cv
import time

cwd = Path(__file__).parent
model_path = cwd / 'hand_landmarker.task'

# Creating a task. A task is essentially a premade appliance by mediapipe to take in raw data from images and spit out annotations.
# We're going to create the hand landmarker task.

# Here are some base settings that do not need to be touched:
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# ==== Starting an instance of Hand landmarker: ====

# expected types for Hand landmarker instance: 
# (result: HandLandMarkerResult, output_image: mp.image, timestamp_ms: int)
def show_results(result, output_image, timestamp_ms):

    # automatically fills the print statement based on the result of our func.
    print('hand land marker result: {}'.format(result))


# Options that should be looked at more deeply
my_options = HandLandmarkerOptions(
    # model path
    base_options=BaseOptions(model_asset_path=model_path),

    # required to run mediapipe on the webcam
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,

    # this essentially says: once an annotation comes through successfully, go back and run the show_result func...
    # which prints what those results were.
    result_callback=show_results)

# ==== Hand landmarker settings are created ====

# To use appliance that will be taking in the raw data we say:
# Using this appliance created with my preferences in mind, do ...
with HandLandmarker.create_from_options(my_options) as landmarker_appliance:

    try:
        webcam = cv.VideoCapture(0)

        while True:
            ret, frame = webcam.read()

            # To be used with mediapipe frames must be in RGB.
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # creating a mediapipe image in RGB format using a Numpy array (openCV output)
            mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # mediapipe requires us to provide a timestamp for each frame so we'll do that here, this will run everytime openCV spits out a frame
            timestamp_ms = int(time.time() * 1000)

            # sending framedata to Handlandmarker for detections:
            landmarker_appliance.detect_async(mediapipe_image, timestamp_ms)

            # cv.waitKey(x) monitors keystrokes and it delays the next openCV frame by X milliseonds. So if you want to save processing power
            # I think upping the milliseconds will save computations because ur making manipulations to less images?
            if cv.waitKey(10) == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")