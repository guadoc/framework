import tensorflow as tf
import tqdm as tq

class Train:
    def __init__(self, model):
        pass
        self.loss = loss = tf.reduce_mean(tf.square(model.outputs- model.labels))
        self.train_step = tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)



    def train(self, controler, train_set, model):

        correct_prediction = tf.equal(tf.argmax(model.outputs,1), tf.argmax(model.labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for bat in tq.tqdm(range(controler.n_batch)):
            batch = train_set.sample( controler.batch_size )
            self.train_step.run(feed_dict={model.inputs: batch['inputs'], model.labels: batch['labels']})
            if bat%100 == 0:
                pass
                #train_accuracy = accuracy.eval(feed_dict={model.inputs: batch[0], model.labels: batch[1]})
                #print("step %d, training accuracy %g"%(bat, train_accuracy))
