import cv2
import numpy as np
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,max_num_hands=1,min_detection_confidence=0.5)

dataset_path = 'data/asl_alphabet_train'
dataset_landmark = 'data_landmarks'

signs = os.listdir(dataset_path)
print(f"Available ASL signs in the dir : {signs}")

for sign in signs :
    os.makedirs(os.path.join(dataset_landmark,sign),exist_ok=True)
    image_files = os.listdir(os.path.join(dataset_path,sign))
    for index,image_name in enumerate(image_files):
        image_path = os.path.join(dataset_path,sign,image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            # The current is not readable. So we're skipping it and continuing
            print(f"Warning : Could not read image #{index}")
            continue
        image_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            # The next assigning statement contains a list of landmarks where each landmark is a point with x,y,z coordinates normalized in the range 0.0-1.0
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for res in hand_landmarks.landmark:
                landmarks.extend([res.x,res.y,res.z])

            npy_path = os.path.join(dataset_landmark,sign,str(index))
            np.save(npy_path,np.array(landmarks))

hands.close()
print("DATA PROCESSING COMPLETED...")
                    