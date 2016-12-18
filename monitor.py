import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import os
import time

class Monitor:
    def __init__(self, opts):
        #training parameters
        self.last_epoch = opts.last_epoch
        self.n_epoch = opts.n_epoch
        self.batch_size = opts.batch_size
        self.checkpoint = opts.checkpoint
        self.epoch = self.last_epoch + 1
        self.n_train_batch_epoch = math.floor(opts.n_data_train / self.batch_size)
        self.n_batch= (self.n_epoch - self.last_epoch) * self.n_train_batch_epoch

        self.epoch_time = time.clock()


        # validation parameters
        self.n_val_batch = 100
        self.val_batch_size = 100

        #log metrics
        self.train_loss = []
        self.val_loss = []


        #monitored metric
        self.train_checkpoint_loss = []
        self.cumulated_train_loss = 0
        self.cumulated_train_accuracy = 0
        self.cumulated_val_loss = 0

        #self.saver = tf.train.Saver()
        self.save_ = None

        #plot
        self.plot_ = {'train_loss':[], 'norm1':[]}
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
        self.train_loss_index = 0
        self.train_norm1_index = 1
        train_metrics = [ \
            tf.reduce_mean(tf.square(model.outputs - model.labels)), \
            tf.reduce_mean(tf.abs(model.outputs - model.labels)) \
            ]
        return train_metrics


    def val_metrics(self, model):
        self.val_loss_index = 0
        val_metrics = [ \
            tf.reduce_mean(tf.square(model.outputs - model.labels)), \
            ]
        return val_metrics



    def train_update(self, metrics):
        self.train_checkpoint_loss.append(metrics[self.train_loss_index])
        self.cumulated_train_loss += metrics[self.train_loss_index]


    def val_update(self, metrics):
        self.cumulated_val_loss += metrics[self.val_loss_index]



    def checking(self, metrics):
        self.plot_['train_loss'].append(np.mean(self.train_checkpoint_loss))
        self.train_checkpoint_loss = []
        self.plot()


    def end_epoch(self, opts, bat, metrics, model):
        #saving model
        model.model_save(os.path.join(opts.expe, opts.model + "_" + str(self.epoch) + ".ckpt"))
        #printing progression
        print('Epoch %d/%gs, testing loss %g, training loss %g'%(self.epoch, time.clock() - self.epoch_time, self.cumulated_val_loss/self.n_val_batch, self.cumulated_train_loss/self.n_train_batch_epoch))
        print('patch') #for tqdm issue
        #updating Logs
        self.train_loss.append(self.cumulated_train_loss/self.n_train_batch_epoch)
        self.val_loss.append(self.cumulated_val_loss/self.n_val_batch)
        self.cumulated_val_loss = 0
        self.cumulated_train_loss = 0
        self.epoch = self.epoch + 1
        self.epoch_time = time.clock()
        #saving Logs --->TODO
