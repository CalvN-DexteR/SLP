import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import pickle

landmark_path = "data_landmarks"

signs = os.listdir(landmark_path)

x_data = []
y_data = []

for sign_index,sign in enumerate(signs):
    sign_path = os.path.join(landmark_path,sign)
    # The os.listdir(sign_path) gives a list of all images under a particular sign(say 'A')
    for image_file in os.listdir(sign_path):
        landmarks = np.load(os.path.join(sign_path,image_file))
        x_data.append(landmarks)
        y_data.append(sign_index)

X = np.array(x_data)
y = np.array(y_data)

print(f"Data Loaded Successfully..\nShape of X : {X.shape}\nShape of Y : {y.shape}")

# Training part

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Initialize the model
model = SVC(kernel='rbf',C=1.0,probability=True)

#Train the model
print('--TRAINING THE MODEL--')
model.fit(X_train,y_train)
print('--MODEL TRAINING COMPLETED--')

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"Model Accuracy {accuracy}")
print("End of Program")