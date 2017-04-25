
# Playground to evaluate without training the encoder
import tensorflow as tf
import numpy as np
import scipy.ndimage as image
import matplotlib.pyplot as plt
from autoencoder_graph import create_autoencoder

np_data = np.load("data/numpy_data.npy")


# Get graph
create_autoencoder()

# Saver operation to restore variables
saver = tf.train.Saver()

# Restore graph
sess = tf.Session()
saver.restore(sess, "saved_models/model5000.ckpt")

print()
print()
print("------------------------------")

# Test the model
image = np_data[:8]
out = sess.run("raw_out:0", feed_dict={"raw_data:0": image})

# Plot some samples

plt.figure()

for i in range(out.shape[0]):
    pueh = (out[i] - np.min(out[i])) / (np.max(out[i]) - np.min(out[i]))
    plt.subplot(2, out.shape[0], i + 1)
    plt.imshow(pueh)
    plt.subplot(2, out.shape[0], i + out.shape[0] + 1)
    plt.imshow(np_data[i]/255)

plt.show()
