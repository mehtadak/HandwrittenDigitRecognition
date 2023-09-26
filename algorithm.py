import tensorflow as tf

mnist = tf.keras.datasets.mnist
#Splitting data into training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalizing the training and test data
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

#Making the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

#Compiling the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', 
              metrics = ['accuracy'])

#Training the model on the training data
model.fit(x_train, y_train, epochs = 3)

#Saving the model in a .model file for possible later use
#Saves time bvy having us not compile the model everytime we want to use it
model.save('handwritten.model')

#Testing the data on the testing data and saving key test metrics
loss, accuracy = model.evaluate(x_test, y_test)

#Printing the important results of the tests to confirm that the model works
print(loss)
print(accuracy)