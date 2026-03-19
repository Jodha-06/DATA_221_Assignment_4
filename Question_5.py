from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


load_data = load_breast_cancer()

X= pd.DataFrame(load_data.data, columns=load_data.feature_names)
y = pd.Series(load_data.target)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42, stratify = y)

Decision_Tree_Model = DecisionTreeClassifier(max_depth=4, random_state=42)
Decision_Tree_Model.fit(X_train, y_train)
y_Decision_Tree_Predictions = Decision_Tree_Model.predict(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Neural_Network_Model = MLPClassifier(max_iter=500, random_state=42)
Neural_Network_Model.fit(X_train_scaled, y_train)
y_Neural_Network_Predictions = Neural_Network_Model.predict(X_test_scaled)

Decision_Tree_Matrix = confusion_matrix(y_test, y_Decision_Tree_Predictions)
Neural_Network_Matrix = confusion_matrix(y_test, y_Neural_Network_Predictions)

print(f"Decision Tree Matrix:\n {Decision_Tree_Matrix}")
print(f"Neural Network Matrix:\n {Neural_Network_Matrix}")

#The neural network performs slightly better than the decision tree, however I would prefer the decision tree for this task, because interpretability is important and in medical applications such as this one understanding the data and the decisions is crucial

# An advantage of the Decision Tree is that it is easy to interpret, but a limitation is that it can overfit and may be less accurate

# An advantage of a Neural Network is that it has higher accuracy and better generalization, but a limitation is that it is harder to interpret