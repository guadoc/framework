from tensorflow.examples.tutorials.mnist import input_data

def get_metadata():
    #meta_data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    meta_data = {}#{'train':0, 'val':1}
    meta_data['train']=1
    meta_data['val']=1
    return meta_data
