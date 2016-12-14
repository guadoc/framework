import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

class Monitor:
    def __init__(self, opts):
        self.last_epoch = opts.last_epoch
        self.n_epoch = opts.n_epoch
        self.batch_size = opts.batch_size

        self.epoch = self.last_epoch + 1
        self.n_batch_epoch = 4
        self.n_exemple_averaging = 100
        self.n_batch_averaging = math.floor(self.n_exemple_averaging / self.batch_size)
        self.n_batch= (self.n_epoch - self.last_epoch) * self.n_batch_epoch

        #self.saver = tf.train.Saver()
        self.save_ = None

        #plot
        self.plot_ = {'train_loss':np.empty([1])}
        plt.ion()
        self.ax = plt.gca()
        self.line, = self.ax.plot([], [])


    def plot(self):
        self.line.set_xdata(range(len(self.plot_['train_loss'])))
        self.line.set_ydata(self.plot_['train_loss'])
        self.ax.relim()
        #self.ax.autoscale_view(True,True,True)
        plt.xlabel('Averaged batches')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.plot()
        plt.pause(0.0000001)


    def metrics(self, model):
        metrics = {'loss':tf.reduce_mean(tf.square(model.outputs- model.labels))}
        return metrics


    def check_n_update(self, train_loss):
        self.plot_['train_loss'] = np.append(self.plot_['train_loss'], train_loss)
        self.plot()


    def end_epoch(self, opts, bat, train_loss, model):
        save_path = self.saver.save(model.sess, "./expe/"+opts.model+"_"+str(self.epoch)+".ckpt")
        print("Model saved in file: %s" % save_path)
        # test and print global statistics(accuracy, time, loss, ...)
        print("Epoch %d, training loss %g"%(self.epoch, train_loss))
        self.epoch = self.epoch + 1
