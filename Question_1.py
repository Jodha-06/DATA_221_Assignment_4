from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

load_data = load_breast_cancer()

X= pd.DataFrame(load_data.data, columns=load_data.feature_names)
y = pd.Series(load_data.target)

print(f"The shape of X is {X.shape}")
print(f"The shape of y is {y.shape}")

print("\nNumber of Samples belonging to each class: ")
print(y.value_counts())

# This dataset contains 569 samples and 30 features
# IN this dataset there are 357 benign  cases (represented by class 1 ) and 212 malignant cases (represented by class 0)

# This dataset is somewhat imbalanced as the benign class has more samples than the malignant class

# Class balance is important as an imbalanced dataset can lead to biased models that favor the class with more samples, therefore negatively impacting the performance of the class with less samples