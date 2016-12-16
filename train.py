import tensorflow as tf
import tqdm as tq
import numpy as np
from test import Test

class Train:
    def __init__(self, monitor, model):
        self.metrics = monitor.train_metrics(model)
        self.lr = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.metrics[monitor.train_loss])
        self.tester = Test(monitor, model)


    def train(self, opts, monitor, train_set, test_set, model):
        for bat in tq.tqdm(range(monitor.n_batch)): # Cest ici le probleme. ;            
            batch_sample = train_set.sample( monitor.batch_size )
            lr = model.lr_schedule(monitor, bat)
            _, metrics = model.sess.run([self.optimizer, self.metrics], feed_dict={self.lr: lr, model.inputs: batch_sample['inputs'], model.labels: batch_sample['labels']})
            monitor.update(opts, metrics, bat)
            if bat%monitor.checkpoint == 0:
                monitor.checking(metrics)
            if bat%monitor.n_batch_epoch == 0:
                metrics = self.tester.valid(monitor, test_set, model)
                monitor.end_epoch(opts, bat, metrics, model)
