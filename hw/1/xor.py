import tensorflow as tf
from tensorflow import keras

# Define the model architecture
model = keras.Sequential()
model.add(keras.layers.Dense(8, input_dim=2, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the input and output data
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# Train the model on the input and output data
model.fit(X, y, epochs=500, batch_size=4)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set the optimization flag to "optimization_type.DEFAULT"
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set the input and output data types to int8
converter.target_spec.supported_types = [tf.int8]


# Provide a representative dataset for the quantization process
def representative_dataset():
    for i in range(len(X)):
        yield [tf.constant(X[i], shape=(1, 2), dtype=tf.float32)]


converter.representative_dataset = representative_dataset

# Convert the model to a TFLite flatbuffer file
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('quantized_model.tflite', 'wb') as f:
    f.write(tflite_model)

