# SARS 2.0 Exposure Predication by Deep Learning
import tensorflow as tf
import numpy as np
days = np.array([1, 2,  3,  4, 5, 6],  dtype=float)
exposures = np.array([139, 224, 291, 324, 440, 473],  dtype=float)
layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])
model.compile(loss='mean_squared_error', 
              optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(days, exposures, epochs=3000, verbose=False)
day=6 # Jan 19, 2020 is the first day reported 139 exposures.
print("{}th day exposures {}".format(day, int(model.predict([day])[0][0])))