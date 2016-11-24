import tensorflow as tf
import tqdm as tq

class Train:
    def __init__(self):
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))


def train(controler):
    for bat in tqdm.tqdm(range(controler.nBatch)):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if bat%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
