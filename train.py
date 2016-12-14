import tensorflow as tf
import tqdm as tq
import numpy as np

class Train:
    def __init__(self, monitor, model):
        self.metrics = monitor.metrics(model)
        self.train_step = tf.train.GradientDescentOptimizer(0.000001).minimize(self.metrics['loss'])


    def train(self, opts, monitor, dataset, model):

        for bat in tq.tqdm(range(monitor.n_batch)):
            batch = dataset.sample( monitor.batch_size )
            self.train_step.run(feed_dict={model.inputs: batch['inputs'], model.labels: batch['labels']})
            if bat%monitor.checkpoint == 0:
                train_loss = self.metrics['loss'].eval(feed_dict={model.inputs: batch['inputs'], model.labels: batch['labels']})
                monitor.check_n_update(train_loss)
            if bat%monitor.n_batch_epoch == 0:
                train_loss = self.metrics['loss'].eval(feed_dict={model.inputs: batch['inputs'], model.labels: batch['labels']})
                #test()
                monitor.end_epoch(opts, bat, train_loss, model)
