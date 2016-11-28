import tensorflow as tf
import tqdm as tq
import numpy as np

class Train:
    def __init__(self, model):
        pass
        self.loss = tf.reduce_mean(tf.square(model.outputs- model.labels))
        self.train_step = tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)



    def train(self, controler, dataset, model):

        correct_prediction = tf.equal(tf.argmax(model.outputs,1), tf.argmax(model.labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print(controler.n_batch_averaging )
        for bat in tq.tqdm(range(10)):#controler.n_batch)):
            batch = dataset.sample( controler.batch_size )
            self.train_step.run(feed_dict={model.inputs: batch['inputs'], model.labels: batch['labels']})
            if bat%controler.n_batch_averaging == 0:
                train_loss = self.loss.eval(feed_dict={model.inputs: batch['inputs'], model.labels: batch['labels']})
                np.append(controler.plot_['train_loss'], train_loss)
                controler.plot()
                print("step %d, training accuracy %g"%(bat, train_loss))
