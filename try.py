import tensorflow as tf
import threading as th
sess = tf.Session()

q = tf.FIFOQueue(8 , "float")
z = tf.placeholder("float", 1)
q_inc = q.enqueue(z[0])
q_dec = q.dequeue()



nBatch = 20
y = 1.


compt = tf.Variable([1.0])
init = tf.global_variables_initializer()
sess.run(init)
print(compt.eval(session=sess))

def inc(coord, y, compt, nBatch):
    while not coord.should_stop():
        compt = compt + [1.]
        a = 1#compt.eval(sess)
        print(a)
        sess.run(q_inc, feed_dict={z:[a]})
        y = y + 1
        #print(compt.eval(sess))
        if y == nBatch:
            coord.request_stop()

def dec(coord):
    while not coord.should_stop():
        a = sess.run(q_dec)
        #print('dec %g'%(a))


coord_inc = tf.train.Coordinator()
coord_dec = tf.train.Coordinator()
n_thread = 4
threads_inc = [th.Thread(target=inc, args =(coord_inc, y, compt, nBatch)) for i in range(n_thread)]
threads_dec = [th.Thread(target=dec, args =(coord_dec,)) for i in range(1)]


for t in threads_inc: t.start()
for t in threads_dec: t.start()

coord_inc.join(threads_inc)
coord_dec.request_stop()
coord_dec.join(threads_dec)
