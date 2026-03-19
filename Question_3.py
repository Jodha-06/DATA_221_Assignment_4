from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

load_data = load_breast_cancer()

X= pd.DataFrame(load_data.data, columns=load_data.feature_names)
y = pd.Series(load_data.target)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42, stratify = y)

Decision_Tree_Model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
Decision_Tree_Model.fit(X_train, y_train)

y_train_prediction = Decision_Tree_Model.predict(X_train)
y_test_prediction = Decision_Tree_Model.predict(X_test)
y_train_accuracy = accuracy_score(y_train, y_train_prediction)
y_test_accuracy = accuracy_score(y_test, y_test_prediction)

print(f"Training Accuracy: {y_train_accuracy}")
print(f"Testing Accuracy: {y_test_accuracy}")

importances = pd.Series(Decision_Tree_Model.feature_importances_, index=X.columns)
print("\nTop 5 Features: ")
print(importances.sort_values(ascending=False).head(5))

#Limiting the maximum depth of the model reduces overfitting by preventing the tree from becoming too complex and from memorizing the training data

# Compared to the unconstrained model for Question 2, the training accuracy decreased slightly however the test accuracy improved, which indicates better generalization

# Feature importance shows which features contibute the most to predictions made by the model. This improves overall interpretability as it helps us understand which variables influence the model