import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

class Controler:
    def __init__(self):
        self.last_epoch = 0
        self.n_epoch = 200
        self.n_batch_epoch = 4
        self.batch_size = 10
        self.n_exemple_averaging = 500
        self.n_batch_averaging = math.floor(self.n_exemple_averaging / self.batch_size)
        self.n_batch= (self.n_epoch - self.last_epoch) * self.n_batch_epoch        
        self.plot_ = {'train_loss':np.empty([1])}
        self.save_ = None

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
