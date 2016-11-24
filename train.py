import tensorflow as tf


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
