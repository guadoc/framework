import tensorflow as tf
sess = tf.Session()
x = tf.Variable([1.0, 2.0])
sess.run(tf.initialize_all_variables())

print(x.eval(session=sess))
x = 2* x
print(x.eval(session=sess))
