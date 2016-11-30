import tensorflow as tf
#from models.baseline import create_model

class Model(object):
    def __init__(self, opts):
        model_path = "models."+opts.model
        module = __import__(model_path)

        create_model = __import__("create_model")
        self.sess = tf.InteractiveSession()
        self.inputs = tf.placeholder(tf.float32, shape=[None, 1])
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        self.outputs = create_model(self)
        self.sess.run(tf.global_variables_initializer())
