import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

CNN_Model = models.Sequential([
    layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

CNN_Model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

CNN_Model.fit(X_train,y_train,epochs=5, verbose=0)

y_probability_prediction = CNN_Model.predict(X_test)
y_prediction = y_probability_prediction.argmax(axis=1)

Confusion_Matrix = confusion_matrix(y_test,y_prediction)
print(f"Confusion_Matrix:\n {Confusion_Matrix}")

misclassified_images = np.where(y_prediction != y_test)[0]

for i in range(3):
    idx = misclassified_images[i]
    plt.imshow(X_test[idx].reshape(28,28))
    plt.title(f"True: {y_test[idx]}, Predicted: {y_prediction[idx]}")
    plt.axis('off')
    plt.show()


# One pattern observed in the missclassification is that similar clothing items are often confused (eg. shirts and t-shirts)

# A realistic way to improve performance is to increase the models complexity by adding for layers or by using data augmentation in order to provide more diverse training examples.