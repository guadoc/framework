import tensorflow as tf
import importlib
#from models.baseline import create_model

class Model(object):
    def __init__(self, opts):
        module = importlib.import_module(opts.model_path+'.'+opts.model)
        self.sess = tf.Session()
        self.inputs = tf.placeholder(tf.float32, shape=[None, 1])
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # with tf.device("/gpu:0"):
        self.outputs = module.create_model(self)
        self.saver = tf.train.Saver(max_to_keep=1000)

        if opts.last_epoch > 0:
            self.model_load("./expe/"+ opts.model + "_" +str(opts.last_epoch)+".ckpt")
        else:
            self.sess.run(tf.global_variables_initializer())


    def model_save(self, path):
        save_path = self.saver.save(self.sess, path)
        print("## Model saved in file: %s" % save_path)

    def model_load(self, path):
        print("-- Loading model from file: %s" % path)
        self.saver.restore(self.sess, path)
        print("## Model loaded from file: %s" % path)
