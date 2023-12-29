import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, D
from getkeys import key_check
import tensorflow as tf
import os

# MODEL_NAME = 'pygta5-car-0.001-alexnetv2-12-epochs.model'
MODEL_NAME = 'pygta5-car-0.001-alexnetv2-8-epochs.model'
# Check if the file exists
if os.path.exists(MODEL_NAME):
    print(f"Model saved successfully at: {MODEL_NAME}")
else:
    print(f"Error: Model could not be saved at {MODEL_NAME}")

# Load the saved model and print summary
loaded_model = tf.keras.models.load_model(MODEL_NAME)
loaded_model.summary()

# Get the appropriate input signature for the model
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)

def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)

def main():
    last_time = time.time()
    moves = [0, 0, 0] 
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    while True:
        if not paused:
            # 800x600 windowed mode
            screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
            print('loop took {} seconds'.format(time.time() - last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160, 120))
            
            prediction = loaded_model.predict(screen.reshape(1, 160, 120, 1))
            moves = list(np.around(prediction)[0])
            print('Predictions:', prediction)  # Print the predictions

            if moves == [1, 0, 0]:
                left()
            elif moves == [0, 1, 0]:
                straight()
            elif moves == [0, 0, 1]:
                right()

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

if __name__ == "__main__":
    main()
