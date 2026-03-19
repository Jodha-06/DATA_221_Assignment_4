import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.datasets import fashion_mnist

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

CNN_Model.fit(X_train,y_train,epochs=15, validation_split=0.1)
test_loss, test_accuracy = CNN_Model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_accuracy}")

# CNNs are prefered over fully connected networks for image data because they are able to preserve spatial relationships between pixels meaning they can automatically learn important patterns.
#This is different from fully connected networks, as they treat each pixel independantly, which leads to a loss of spatial information and makes them less effective for image related tasks