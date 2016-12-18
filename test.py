import tensorflow as tf
import tqdm as tq

class Test:
    def __init__(self, monitor, model):
        self.metrics = monitor.val_metrics(model)        

    def valid(self, monitor, dataset, model):
        self.update_model(model)
        for bat in tq.tqdm(range(monitor.n_val_batch)):
            batch_sample = dataset.sample( monitor.batch_size )
            metrics = model.sess.run(self.metrics, feed_dict={model.inputs: batch_sample['inputs'], model.labels: batch_sample['labels']})
            monitor.val_update(metrics)


    def update_model(self, model):
        self.model = model
