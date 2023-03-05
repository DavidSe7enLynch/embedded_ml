### Tweak Model

- original

  ```python
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(100, activation='relu'))
  model.add(tf.keras.layers.Dense(50, activation='relu'))
  # model.add(tf.keras.layers.Dense(25, activation='relu'))
  model.add(tf.keras.layers.Dense(NUM_GESTURES,
                                  activation='softmax'))
  model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
  ```

  predictions =
   [[0.02  0.98 ]
   [0.    1.   ]
   [1.    0.   ]
   [0.    1.   ]
   [0.666 0.334]
   [0.001 0.999]
   [0.    1.   ]
   [0.999 0.001]]
  actual =
   [[0. 1.]
   [0. 1.]
   [1. 0.]
   [0. 1.]
   [0. 1.]
   [0. 1.]
   [0. 1.]
   [1. 0.]]

  accuracy = 7/8 = 0.875

  

- increased

  ```python
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(100, activation='relu'))
  model.add(tf.keras.layers.Dense(50, activation='relu'))
  model.add(tf.keras.layers.Dense(25, activation='relu'))
  model.add(tf.keras.layers.Dense(NUM_GESTURES,
                                  activation='softmax'))
  model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
  ```

  predictions =
   [[0.001 0.999]
   [0.    1.   ]
   [1.    0.   ]
   [0.    1.   ]
   [0.433 0.567]
   [0.    1.   ]
   [0.    1.   ]
   [1.    0.   ]]
  actual =
   [[0. 1.]
   [0. 1.]
   [1. 0.]
   [0. 1.]
   [0. 1.]
   [0. 1.]
   [0. 1.]
   [1. 0.]]

  accuracy = 1

  

- decreased:

  ```python
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(50, activation='relu'))
  model.add(tf.keras.layers.Dense(25, activation='relu'))
  # model.add(tf.keras.layers.Dense(25, activation='relu'))
  model.add(tf.keras.layers.Dense(NUM_GESTURES,
                                  activation='softmax'))
  model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
  ```

  predictions =
   [[0.019 0.981]
   [0.    1.   ]
   [1.    0.   ]
   [0.    1.   ]
   [0.525 0.475]
   [0.    1.   ]
   [0.    1.   ]
   [1.    0.   ]]
  actual =
   [[0. 1.]
   [0. 1.]
   [1. 0.]
   [0. 1.]
   [0. 1.]
   [0. 1.]
   [0. 1.]
   [1. 0.]]

  accuracy = 0.875