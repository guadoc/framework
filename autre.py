import tensorflow as tf
import threading as th
from data.data import Data
from data.metadata import get_metadata
from config import init_config
import numpy as np
import time

opts = init_config()

#initialization of datasets
meta_data = get_metadata()
train_set = Data(meta_data['train'])
n_thread = 1
nBatch = 128
batch_size = 253*256

sess = tf.Session()
data_q = tf.FIFOQueue(10000 , [tf.float32], shapes=[batch_size, 1])
compt_q = tf.FIFOQueue(1 , tf.int64)

compteur = tf.placeholder(tf.int64, [])
data = tf.placeholder(tf.float32, [batch_size, 1])

data_inc = data_q.enqueue(data)
data_dec = data_q.dequeue()
compt_inc = compt_q.enqueue(compteur)
compt_dec = compt_q.dequeue()

sess.run(compt_inc, feed_dict={compteur:0})
run_options = tf.RunOptions(timeout_in_ms=100)


def data_fill(coord, nBatch, dataset, batch_size):
    pass
    while not coord.should_stop():
        try:
            a = sess.run(compt_dec, options=run_options)
            a = a + 1
            if a >= nBatch:
                coord.request_stop()
            print(a)
            inputs = dataset.sample( batch_size )['inputs']
            sess.run(compt_inc, feed_dict={compteur:a})
            #sess.run(data_inc, feed_dict={data:inputs})
        except tf.errors.DeadlineExceededError:
            print("empty")
            pass

def data_use(coord):
    while not coord.should_stop():#
        a = sess.run(compt_dec, options=run_options)
        print(a)
        if a >= nBatch:
            coord.request_stop()
        sess.run(compt_inc, feed_dict={compteur:a})
        u = sess.run(data_dec)
        #print(np.shape(u))

coord_inc = tf.train.Coordinator()
#coord_dec = tf.train.Coordinator()

threads_inc = [th.Thread(target=data_fill, args =(coord_inc, nBatch, train_set, batch_size)) for i in range(n_thread)]
#threads_dec = [th.Thread(target=data_use, args =(coord_dec,)) for i in range(n_thread)]

begin = time.clock()

for t in threads_inc: t.start()
#for t in threads_dec: t.start()

coord_inc.join(threads_inc)
#coord_dec.request_stop()
#coord_dec.join(threads_dec)

end = time.clock()

print(end - begin)
