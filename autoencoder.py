# Typical setup to include TensorFlow.
import tensorflow as tf
import numpy as np
import scipy.ndimage as image
import matplotlib.pyplot as plt

np_data = np.load("data/numpy_data.npy")

#plt.imshow(np_data[5]/255)
#plt.draw()


# Define graph
with tf.device("/gpu:0"):
    k = 64*64*3
    z = 40
    raw_data = tf.placeholder(tf.float32, [None, 64, 64, 3])
    x = tf.reshape(raw_data, [tf.shape(raw_data)[0], k])
    W1 = tf.Variable(tf.truncated_normal([k, z], stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal([z], stddev=0.1))

    h = tf.nn.xw_plus_b(x, W1, b1) 

    # Decoder

    W2 = tf.Variable(tf.truncated_normal([z, k], stddev=0.1))
    b2 = tf.Variable(tf.truncated_normal([k], stddev=0.1))

    y = tf.nn.xw_plus_b(h, W2, b2)
    raw_out = tf.reshape(y, [tf.shape(raw_data)[0], 64, 64, 3])

# Optimizer 
err = tf.reduce_mean((x - y) ** 2)
train_step = tf.train.AdamOptimizer().minimize(err)

# Do the training
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
print()
print()
print("------------------------------")

batch_size = 200
for i in range(5000):
    idx = np.random.randint(0, np_data.shape[0], batch_size)
    data_batch = np_data[idx]

    for j in range(1):
        sess.run(train_step, feed_dict={raw_data: data_batch})

    print(i)

# Yeeey we have a trained model, let's test it
image = np_data[:8]
out = sess.run(raw_out, feed_dict={raw_data: image})

plt.figure()

for i in range(out.shape[0]):
    pueh = (out[i] - np.min(out[i])) / (np.max(out[i]) - np.min(out[i]))
    plt.subplot(2, out.shape[0], i + 1)
    plt.imshow(pueh)
    plt.subplot(2, out.shape[0], i + out.shape[0] + 1)
    plt.imshow(np_data[i]/255)

plt.show()
