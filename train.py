import tensorflow as tf
import tqdm as tq
import numpy as np

class Train:
    def __init__(self, monitor, model):
        self.metrics = monitor.train_metrics(model)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.metrics[monitor.train_loss])


    def train(self, opts, monitor, dataset, model):
        for bat in tq.tqdm(range(monitor.n_batch)):
            batch = dataset.sample( monitor.batch_size )
            _, metrics = model.sess.run([self.optimizer, self.metrics], feed_dict={model.inputs: batch['inputs'], model.labels: batch['labels']})
            monitor.update(metrics)
            if bat%monitor.checkpoint == 0:
                monitor.checking(metrics)
            if bat%monitor.n_batch_epoch == 0:
                pass
                #test()
                #monitor.end_epoch(opts, bat, metrics, model)
