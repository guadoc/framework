import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import os

class Monitor:
    def __init__(self, opts):
        #training parameters
        self.last_epoch = opts.last_epoch
        self.n_epoch = opts.n_epoch
        self.batch_size = opts.batch_size
        self.checkpoint = opts.checkpoint
        self.epoch = self.last_epoch + 1
        self.n_batch_epoch = math.floor(opts.n_data_train / self.batch_size)
        self.n_batch= (self.n_epoch - self.last_epoch) * self.n_batch_epoch


        # validation parameters
        self.n_val_batch = 100
        self.val_batch_size = 100

        #useful metrics indexes
        self.train_loss = 0
        self.norm1 = 1

        #monitored metric
        self.train_checkpoint_loss = np.zeros(opts.checkpoint)
        self.train_epoch_cumulated_loss = 0
        self.train_epoch_cumulated_accuracy = 0

        #self.saver = tf.train.Saver()
        self.save_ = None

        #plot
        self.plot_ = {'train_loss':np.array([]), 'norm1':np.array([])}
        fig = plt.figure()
        self.ax = fig.add_subplot(111)


    def plot(self):
        self.ax.clear()
        plt.hold(True)
        plt.xlabel('Averaged batches')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.plot(range(len(self.plot_['train_loss'])), self.plot_['train_loss'])
        plt.pause(0.0000001)


    def train_metrics(self, model):
        train_metrics = [ \
            tf.reduce_mean(tf.square(model.outputs - model.labels)), \
            tf.reduce_mean(tf.abs(model.outputs - model.labels)) \
            ]
        return train_metrics


    def val_metrics(self, model):
        val_metrics = [ \
            tf.reduce_mean(tf.square(model.outputs - model.labels)), \
            tf.reduce_mean(tf.abs(model.outputs - model.labels)) \
            ]
        return val_metrics



    def update(self, opts, metrics, batch):
        self.train_checkpoint_loss[batch % opts.checkpoint] = metrics[self.train_loss]


    def checking(self, metrics):
        self.plot_['train_loss'] = np.append(self.plot_['train_loss'], np.mean(self.train_checkpoint_loss))
        #self.plot_['norm1'] = np.append(self.plot_['norm1'], metrics[self.norm1]
        self.plot()


    def end_epoch(self, opts, bat, metrics, model):
        model.model_save(os.path.join(opts.expe, opts.model + "_" + str(self.epoch) + ".ckpt"))
        print('Epoch %d, testing loss %g'%(self.epoch, metrics[1]))
        print('patch') #for tqdm issue
        self.epoch = self.epoch + 1
