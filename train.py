import tensorflow as tf
import tqdm as tq
import numpy as np

class Train:
    def __init__(self, controler, model):
        self.metrics = controler.metrics(model)
        self.train_step = tf.train.GradientDescentOptimizer(0.001).minimize(self.metrics['loss'])



    def train(self, controler, dataset, model):
        #correct_prediction = tf.equal(tf.argmax(model.outputs,1), tf.argmax(model.labels,1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        for bat in tq.tqdm(range(controler.n_batch)):
            batch = dataset.sample( controler.batch_size )
            self.train_step.run(feed_dict={model.inputs: batch['inputs'], model.labels: batch['labels']})
            if bat%controler.n_batch_averaging == 0:
                train_loss = self.metrics['loss'].eval(feed_dict={model.inputs: batch['inputs'], model.labels: batch['labels']})
                update_controle(controler, train_loss)
            if bat%controler.n_batch_epoch == 0:
                train_loss = self.metrics['loss'].eval(feed_dict={model.inputs: batch['inputs'], model.labels: batch['labels']})
                end_epoch(controler, bat, train_loss)


def update_controle(controler, train_loss):
    controler.plot_['train_loss'] = np.append(controler.plot_['train_loss'], train_loss)
    controler.plot()

def end_epoch(controler, bat, train_loss):
        print("step %d, training accuracy %g"%(bat, train_loss))
