import tensorflow as tf
import tqdm as tq

class Test:
    def __init__(self, model):
        pass
        self.loss = loss = tf.reduce_mean(tf.square(model.outputs- model.labels))
        self.train_step = tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)
        correct_prediction = tf.equal(tf.argmax(model.outputs,1), tf.argmax(model.labels,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    def test(self, controler, dataset, model):
        batch = dataset.sample(controler.batch_size)
        test_loss = self.loss.eval(feed_dict={model.inputs: batch['inputs'], model.labels: batch['labels']})
        #self.train_step.run(feed_dict={model.inputs: batch['inputs'], model.labels: batch['labels']})
        print("step %d, training accuracy %g"%(1, test_loss))
