import cv2
import mediapipe as mp
import numpy as np
import os

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():   # Here webcam is known as 'cap'
        ret,frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # To mirror the image, so that the hands are actual and not swapped
        frame = cv2.flip(frame, 1)

        # This image is displayed on the output screen, to avoid previous faced colour issues.
        display_image = frame.copy()

        #Since mediapipe accepts only in RGB format, converting the frame from BGR(given by OpenCV) to RGB
        image_mediapipe = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


        image_mediapipe.flags.writeable = False

        # Processing the image using the 'Holistic' model.
        results = holistic.process(image_mediapipe)

        # image_mediapipe.flags.writeable = True
        # image_mediapipe = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        # Drawing the landmarks
        mp_drawing.draw_landmarks(display_image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(display_image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)

        cv2.imshow("Sign Language Recognition",display_image)

        # The screen will close upon pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()