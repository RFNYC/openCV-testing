import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import cv2 as cv
import time

cwd = Path(__file__).parent
model_path = cwd / 'gesture_recognizer.task'
image_path = str(cwd / 'thumbsup.jpg')

# Base settings
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def show_results(result, output_image, timestamp_ms):
    print('hand land marker result: {}'.format(result))


# Options that should be looked at more deeply
my_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=show_results)


with GestureRecognizer.create_from_options(my_options) as gesture_appliance:

    try:
            my_image = cv.imread(image_path)

            rgb_frame = cv.cvtColor(my_image, cv.COLOR_BGR2RGB)

            # creating a mediapipe image in RGB format using a Numpy array (openCV output)
            mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # mediapipe requires us to provide a timestamp for each frame so we'll do that here, this will run everytime openCV spits out a frame
            timestamp_ms = int(time.time() * 1000)

            gesture_appliance.recognize_async(mediapipe_image, timestamp_ms)
            
            cv.imshow('display window', my_image)

            k = cv.waitKey(0)

    except Exception as e:
        print(f"Error: {e}")