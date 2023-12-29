import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

def keys_to_output(keys):
    output = [0, 0, 0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output

file_name = 'training_data.npy'
if os.path.isfile(file_name):
    print("File exists, loading previous data!")
    loaded_data = list(np.load(file_name, allow_pickle=True))
    training_data = [item[0] for item in loaded_data] if loaded_data else []
else: 
    print('File does not exist, starting fresh')
    training_data = []

def main():
    try:
        for i in list(range(4))[::-1]:
            print(i+1)
            time.sleep(1)
            
        while True:
            print("Loop executed")
            # 800x600 windowed mode
            screen = grab_screen(region=(0, 40, 800, 640))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (80, 60))
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append(screen)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            if len(training_data) % 500 == 0:
                print(f'Saving data, current length: {len(training_data)}')
                np.save(file_name, np.array(training_data), allow_pickle=True)  # Save as a numpy array

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Saving data before exiting.")
        np.save(file_name, np.array(training_data), allow_pickle=True)  # Save data before exiting
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
