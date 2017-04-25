# Playground to evaluate without training the encoder
import tensorflow as tf
import numpy as np
import scipy.ndimage as image
import matplotlib.pyplot as plt
from autoencoder_graph import create_autoencoder
import sys
import os

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

# If we have command line arguments, do fun stuff!
if len(sys.argv) > 0:
    url = sys.argv[1]
    os.system("convert -resize 64x64\! {0} /tmp/das_pic.jpg".format(url))
    pic = image.imread("/tmp/das_pic.jpg")

    plt.figure()
    plt.imshow(pic/255)

    pic_in = pic.reshape([1, 64, 64, 3])
    out = sess.run("raw_out:0", feed_dict={"raw_data:0": pic_in})
    plt.figure()
    pueh = (out[0] - np.min(out[0])) / (np.max(out[0]) - np.min(out[0]))
    plt.imshow(pueh)

# Test the model
images = np_data[:8]
out = sess.run("raw_out:0", feed_dict={"raw_data:0": images})

# Plot some samples

plt.figure()

for i in range(out.shape[0]):
    pueh = (out[i] - np.min(out[i])) / (np.max(out[i]) - np.min(out[i]))
    plt.subplot(2, out.shape[0], i + 1)
    plt.imshow(pueh)
    plt.subplot(2, out.shape[0], i + out.shape[0] + 1)
    plt.imshow(np_data[i]/255)

plt.show()
