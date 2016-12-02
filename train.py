import tensorflow as tf
import tqdm as tq
import numpy as np

class Train:
    def __init__(self, controler, model):
        self.metrics = controler.metrics(model)
        self.train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(self.metrics['loss'])



    def train(self, controler, dataset, model):
        saver = tf.train.Saver()
        for bat in tq.tqdm(range(controler.n_batch)):
            batch = dataset.sample( controler.batch_size )
            self.train_step.run(feed_dict={model.inputs: batch['inputs'], model.labels: batch['labels']})
            if bat%controler.n_batch_averaging == 0:
                train_loss = self.metrics['loss'].eval(feed_dict={model.inputs: batch['inputs'], model.labels: batch['labels']})
                update_controle(controler, train_loss)
            if bat%controler.n_batch_epoch == 0:
                train_loss = self.metrics['loss'].eval(feed_dict={model.inputs: batch['inputs'], model.labels: batch['labels']})
                end_epoch(controler, bat, train_loss, model, saver)


def update_controle(controler, train_loss):
    controler.plot_['train_loss'] = np.append(controler.plot_['train_loss'], train_loss)
    controler.plot()

def end_epoch(controler, bat, train_loss, model, saver):
    save_path = saver.save(model.sess, "./expe/model_"+str(controler.epoch)+".ckpt")
    print("Model saved in file: %s" % save_path)
    # test and print global statistics(accuracy, time, loss, ...)
    print("step %d, training accuracy %g"%(bat, train_loss))
    controler.epoch = controler.epoch + 1
