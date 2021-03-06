# Typical setup to include TensorFlow.
import tensorflow as tf
import numpy as np
import scipy.ndimage as image
import matplotlib.pyplot as plt
from autoencoder_graph import create_autoencoder

np_data = np.load("data/numpy_data.npy")

#plt.imshow(np_data[5]/255)
#plt.draw()


# Get graph
create_autoencoder()

# Saver operation to store variables
saver = tf.train.Saver()

# Do the training
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
print()
print()
print("------------------------------")

batch_size = 200
iters = 5000
for i in range(iters):
    idx = np.random.randint(0, np_data.shape[0], batch_size)
    data_batch = np_data[idx]

    for j in range(1):
        sess.run("train_step", feed_dict={"raw_data:0": data_batch})

    print(i)

# Save trained model
save_path = saver.save(sess, "./saved_models/model{0}.ckpt".format(iters))
print("Saved model in {0}".format(save_path))

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
