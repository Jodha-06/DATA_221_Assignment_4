import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

load_data = load_breast_cancer()

X= pd.DataFrame(load_data.data, columns=load_data.feature_names)
y = pd.Series(load_data.target)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42, stratify = y)

Decision_Tree_Model = DecisionTreeClassifier(criterion="entropy", random_state=42)
Decision_Tree_Model.fit(X_train, y_train)

y_train_prediction = Decision_Tree_Model.predict(X_train)
y_test_prediction = Decision_Tree_Model.predict(X_test)
y_train_accuracy = accuracy_score(y_train, y_train_prediction)
y_test_accuracy = accuracy_score(y_test, y_test_prediction)



print(f"Training Accuracy: {y_train_accuracy}")
print(f"Testing Accuracy: {y_test_accuracy}")

#Entropy measures the level of uncertainty in the dataset. A value of 0 indicates that there is a completely "certain" node. The decision tree splits the data to reduce entropy and maximize information gain

# The training accuracy is 1.0 while the testing accuracy is 0.91228. This indicates overfitting, meaning that the model has become too familiar with the training data and does not generalize as effectively to new data
