import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

class Monitor:
    def __init__(self, opts):
        self.last_epoch = opts.last_epoch
        self.n_epoch = opts.n_epoch
        self.batch_size = opts.batch_size
        self.checkpoint = opts.checkpoint

        self.epoch = self.last_epoch + 1
        self.n_batch_epoch = math.floor(opts.n_data_train / self.batch_size)
        self.n_batch= (self.n_epoch - self.last_epoch) * self.n_batch_epoch

        #useful metrics indexes
        self.train_loss = 0
        self.norm1 = 1

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

    def update(self, metrics):
        self.plot_['train_loss'] = np.append(self.plot_['train_loss'], metrics[self.train_loss])
        self.plot_['norm1'] = np.append(self.plot_['norm1'], metrics[self.norm1])

    def checking(self, metrics):
        #self.update(metrics)
        self.plot()


    def end_epoch(self, opts, bat, train_loss, model):
        model.model_save("./expe/"+opts.model+"_"+str(self.epoch)+".ckpt")
        # test and print global statistics(accuracy, time, loss, ...)
        print("Epoch %d, training loss %g"%(self.epoch, train_loss))
        self.epoch = self.epoch + 1
