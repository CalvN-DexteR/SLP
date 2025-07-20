import cv2
import mediapipe as mp
import numpy as np
import pickle,pyttsx3

from TTS import speak

# Load the trained model 'sign_modelv1.pkl' and label map
with open('models/sign_modelv1.pkl','rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    label_map = saved_data['sign-map']

# Initializing TTS Engine
engine = pyttsx3.init()

# Initilaize MediaPipe Hands Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

predictions = []
sentence = []
threshold = 0.95

# with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

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


        # image_mediapipe.flags.writeable = False

        # Processing the image using the 'Hands' model.
        results = hands.process(image_mediapipe)

        # PREDICTION LOGIC
        if results.multi_hand_landmarks:
            # print(results.multi_hand_landmarks)
            # print(type(results.multi_hand_landmarks))
            hand_landmarks = results.multi_hand_landmarks[0]

        # image_mediapipe.flags.writeable = True
        # image_mediapipe = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        # Drawing the landmarks
            mp_drawing.draw_landmarks(display_image,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            # mp_drawing.draw_landmarks(display_image,results.right_hand_landmarks,mp_hands.HAND_CONNECTIONS)

            # Extract the 63 landmark coordinates into a flat list
            landmarks = []
            for res in hand_landmarks.landmark:
                landmarks.extend([res.x,res.y,res.z])
            
            # Make prediction
            prediction = model.predict([landmarks])[0]
            # Get the confidence score
            confidence_scores = model.predict_proba([landmarks])[0]

            # Get the predicted sign-name and its confidence score
            predicted_sign = label_map[prediction]
            prediction_confidence = np.max(confidence_scores)

            predictions.append(predicted_sign)

            if  np.all(np.array(predictions[-10:]) == predicted_sign) and prediction_confidence > threshold :
                #  if len(sentence) == 0 or sentence[-1]!=predicted_sign:
                      sentence.append(predicted_sign)
                      print(f"Sign Confirment : {predicted_sign}")
                      speak(predicted_sign)
                      predictions = []
            if len(predictions) > 20:
                 predictions = predictions[10:]

            else:
                 predictions = []


            if 'predicted_sign' in locals() and results.multi_hand_landmarks:
                 cv2.putText(display_image,f"{predicted_sign}({prediction_confidence : .2f})",(15,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                 cv2.putText(display_image,''.join(sentence),(15,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
                 cv2.imshow("Sign Language Recognition",display_image)

            # The screen will close upon pressing 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


print("Closing SLP")
cap.release()
cv2.destroyAllWindows()
hands.close()