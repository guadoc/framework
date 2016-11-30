import tensorflow as tf

def create_model(mod):
    W1 = tf.Variable(tf.zeros([1,100]))
    b1 = tf.Variable(tf.zeros([100]))
    h_fc1 = tf.nn.relu(tf.matmul(mod.inputs, W1) + b1)
    W2 = tf.Variable(tf.zeros([100,1]))
    b2 = tf.Variable(tf.zeros([1]))
    outputs = tf.matmul(h_fc1, W2) + b2
    return outputs
