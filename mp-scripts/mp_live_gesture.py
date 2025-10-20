import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import cv2 as cv
import time


# ==== NOTE: ====

# PRETTY MUCH ALL YOUR COMMENTS TO UNDERSTAND THIS SRIPT WAS WRITTEN FOR HAND LAND MARKER
# IF YOU CANT REMEMBER WHAT HALF THIS SHIT DOES GO BACK AND READ THAT THROUGH CAREFULLY!

cwd = Path(__file__).parent
model_path = cwd / 'gesture_recognizer.task'


# Base settings
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# ==== Starting an instance of gesture thingamabob: ====

def print_result(result, output_image, timestamp_ms):
    print('gesture recognition result: {}'.format(result))


# Options that should be looked at more deeply
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    # required for webcam
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)


# ==== gesture settings are created ====

with GestureRecognizer.create_from_options(options) as gesture_appliance:

    try:
        webcam = cv.VideoCapture(0)

        while True:
            ret, frame = webcam.read()

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)

            gesture_appliance.recognize_async(mediapipe_image, timestamp_ms)

            # cv.waitKey(x) monitors keystrokes and it delays the next openCV frame by X milliseonds. So if you want to save processing power
            # I think upping the milliseconds will save computations because ur making manipulations to less images?
            if cv.waitKey(10) == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")