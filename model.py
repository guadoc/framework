import tensorflow as tf
import importlib
#from models.baseline import create_model

class Model(object):
    def __init__(self, opts):
        module = importlib.import_module(opts.model_path+'.'+opts.model)
        self.sess = tf.InteractiveSession()
        self.inputs = tf.placeholder(tf.float32, shape=[None, 1])
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        self.outputs = module.create_model(self)
        self.sess.run(tf.global_variables_initializer())
