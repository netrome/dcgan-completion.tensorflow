# Defines the graph for the autoencoder
import tensorflow as tf


# Define graph
def create_autoencoder():
    k = 64*64*3
    z = 40
    raw_data = tf.placeholder(tf.float32, [None, 64, 64, 3], name="raw_data")
    x = tf.reshape(raw_data, [tf.shape(raw_data)[0], k], name="x")
    W1 = tf.Variable(tf.truncated_normal([k, z], stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal([z], stddev=0.1))

    h = tf.nn.xw_plus_b(x, W1, b1) 

    # Decoder

    W2 = tf.Variable(tf.truncated_normal([z, k], stddev=0.1))
    b2 = tf.Variable(tf.truncated_normal([k], stddev=0.1))

    y = tf.nn.xw_plus_b(h, W2, b2, name="y")
    raw_out = tf.reshape(y, [tf.shape(raw_data)[0], 64, 64, 3], name="raw_out")

    # Optimizer 
    err = tf.reduce_mean((x - y) ** 2)
    train_step = tf.train.AdamOptimizer().minimize(err, name="train_step")

