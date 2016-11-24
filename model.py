import tensorflow as tf

class Model(object):
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.inputs = tf.placeholder(tf.float32, shape=[None, 1])
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        W1 = tf.Variable(tf.zeros([1,100]))
        b1 = tf.Variable(tf.zeros([100]))

        h_fc1 = tf.nn.relu(tf.matmul(self.inputs, W1) + b1)

        W2 = tf.Variable(tf.zeros([100,1]))
        b2 = tf.Variable(tf.zeros([1]))

        self.outputs = tf.matmul(h_fc1, W2) + b2
        self.sess.run(tf.initialize_all_variables())
