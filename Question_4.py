from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

load_data = load_breast_cancer()

X= pd.DataFrame(load_data.data, columns=load_data.feature_names)
y = pd.Series(load_data.target)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42, stratify = y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Neural_Network_Model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
Neural_Network_Model.fit(X_train_scaled, y_train)

y_train_prediction = Neural_Network_Model.predict(X_train_scaled)
y_test_prediction = Neural_Network_Model.predict(X_test_scaled)
y_train_accuracy = accuracy_score(y_train, y_train_prediction)
y_test_accuracy = accuracy_score(y_test, y_test_prediction)

print(f"Training Accuracy: {y_train_accuracy}")
print(f"Testing Accuracy: {y_test_accuracy}")

#Feature scaling is necessary for neural networks becasue neural networks rely on gradient descent, meaning if features are on different scales the learning process will become slow or unstable

# An epoch represents one complete pass through the entire training dataset