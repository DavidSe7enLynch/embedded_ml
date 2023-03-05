import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import os


# class GestureModel:
#     def __init__(self):

DIRECTORY = "data_oldboard/"


def plot_raw(filename):
    df = pd.read_csv(DIRECTORY + filename)
    index = range(1, len(df['aX']) + 1)
    plt.rcParams["figure.figsize"] = (20, 10)

    plt.plot(index, df['aX'], 'g.', label='x', linestyle='solid', marker=',')
    plt.plot(index, df['aY'], 'b.', label='y', linestyle='solid', marker=',')
    plt.plot(index, df['aZ'], 'r.', label='z', linestyle='solid', marker=',')
    plt.title("Acceleration")
    plt.xlabel("Sample #")
    plt.ylabel("Acceleration (G)")
    plt.legend()
    plt.show()

    plt.plot(index, df['gX'], 'g.', label='x', linestyle='solid', marker=',')
    plt.plot(index, df['gY'], 'b.', label='y', linestyle='solid', marker=',')
    plt.plot(index, df['gZ'], 'r.', label='z', linestyle='solid', marker=',')
    plt.title("Gyroscope")
    plt.xlabel("Sample #")
    plt.ylabel("Gyroscope (deg/sec)")
    plt.legend()
    plt.show()


# plot_raw("Cutting_HuangR.csv")
# plot_raw("Washing_HuangR.csv")


# Set a fixed random seed value, for reproducibility, this will allow us to get
# the same random numbers each time the notebook is run
SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

# the list of gestures that data is available for
GESTURES = [
    "Cutting_HuangR",
    "Washing_HuangR",
]

SAMPLES_PER_GESTURE = 110

NUM_GESTURES = len(GESTURES)

# create a one-hot encoded matrix that is used in the output
ONE_HOT_ENCODED_GESTURES = np.eye(NUM_GESTURES)


def generate_dataset():
    inputs = []
    outputs = []
    # read each csv file and push an input and output
    for gesture_index in range(NUM_GESTURES):
        gesture = GESTURES[gesture_index]
        print(f"Processing index {gesture_index} for gesture '{gesture}'.")

        output = ONE_HOT_ENCODED_GESTURES[gesture_index]

        df = pd.read_csv(DIRECTORY + gesture + ".csv")

        # calculate the number of gesture recordings in the file
        num_recordings = int(df.shape[0] / SAMPLES_PER_GESTURE)

        print(f"\tThere are {num_recordings} recordings of the {gesture} gesture.")

        for i in range(num_recordings):
            tensor = []
            for j in range(SAMPLES_PER_GESTURE):
                index = i * SAMPLES_PER_GESTURE + j
                # normalize the input data, between 0 to 1:
                # - acceleration is between: -4 to +4
                # - gyroscope is between: -2000 to +2000
                tensor += [
                    (df['aX'][index] + 4) / 8,
                    (df['aY'][index] + 4) / 8,
                    (df['aZ'][index] + 4) / 8,
                    (df['gX'][index] + 2000) / 4000,
                    (df['gY'][index] + 2000) / 4000,
                    (df['gZ'][index] + 2000) / 4000
                ]
            inputs.append(tensor)
            outputs.append(output)
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    return inputs, outputs


inputs, outputs = generate_dataset()


# Randomize the order of the inputs, so they can be evenly distributed for training, testing, and validation
# https://stackoverflow.com/a/37710486/2020087
num_inputs = len(inputs)
randomize = np.arange(num_inputs)
np.random.shuffle(randomize)

# Swap the consecutive indexes (0, 1, 2, etc) with the randomized indexes
inputs = inputs[randomize]
outputs = outputs[randomize]

# Split the recordings (group of samples) into three sets: training, testing and validation
TRAIN_SPLIT = int(0.6 * num_inputs)
TEST_SPLIT = int(0.2 * num_inputs + TRAIN_SPLIT)

inputs_train, inputs_test, inputs_validate = np.split(inputs, [TRAIN_SPLIT, TEST_SPLIT])
outputs_train, outputs_test, outputs_validate = np.split(outputs, [TRAIN_SPLIT, TEST_SPLIT])

print("Data set randomization and splitting complete.")

"""## Build & Train the Model

Build and train a [TensorFlow](https://www.tensorflow.org) model using the high-level [Keras](https://www.tensorflow.org/guide/keras) API.
"""

# build the model and train it
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(100, activation='relu'))  # relu is used for performance
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(NUM_GESTURES,
                                activation='softmax'))  # softmax is used, because we only expect one gesture to occur per input
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = model.fit(inputs_train, outputs_train, epochs=300, batch_size=1,
                    validation_data=(inputs_validate, outputs_validate))

"""## Verify 

Graph the models performance vs validation.

### Graph the loss

Graph the loss to see when the model stops improving.
"""

# increase the size of the graphs. The default size is (6,4).
plt.rcParams["figure.figsize"] = (20, 10)

# graph the loss, the model above is configure to use "mean squared error" as the loss function
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g.', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


"""### Run with Test Data
Put our test data into the model and plot the predictions

"""

# use the model to predict the test inputs
predictions = model.predict(inputs_test)

# print the predictions and the expected ouputs
print("predictions =\n", np.round(predictions, decimals=3))
print("actual =\n", outputs_test)

# Plot the predictions along with to the test data
# plt.clf()
# plt.title('Training data predicted vs actual values')
# plt.plot(inputs_test, outputs_test, 'b.', label='Actual')
# plt.plot(inputs_test, predictions, 'r.', label='Predicted')
# plt.show()

"""# Convert the Trained Model to Tensor Flow Lite

The next cell converts the model to TFlite format. The size in bytes of the model is also printed out.
"""

# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to disk
open("gesture_model.tflite", "wb").write(tflite_model)

basic_model_size = os.path.getsize("gesture_model.tflite")
print("Model is %d bytes" % basic_model_size)

"""## Encode the Model in an Arduino Header File 

The next cell creates a constant byte array that contains the TFlite model. Import it as a tab with the sketch below.
"""

# !echo "const unsigned char model[] = {" > /content/model.h
# !cat gesture_model.tflite | xxd -i      >> /content/model.h
# !echo "};"                              >> /content/model.h


# model_h_size = os.path.getsize("model.h")
# print(f"Header file, model.h, is {model_h_size:,} bytes.")
# print("\nOpen the side panel (refresh if needed). Double click model.h to download the file.")
#
# """# Classifying IMU Data
#
# Now it's time to switch back to the tutorial instructions and run our new model on the Arduino Nano 33 BLE Sense to classify the accelerometer and gyroscope data.
#
# """
