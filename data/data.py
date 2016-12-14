import numpy as np

class Data:
    def __init__(self, meta_data):
        self.size = 0
        self.data = meta_data
        self.dim = [1]

    def sample(self, quantity):
        batch = {'inputs':None, 'labels':None}
        batch['inputs'] = np.random.random_sample(quantity).reshape(quantity,1)
        batch['labels'] = np.exp((batch['inputs']))
        return batch

    def get_exemple(self):
        pass

    def get_next_batch(self, size):
        #batch = self.data.next_batch(size)
        return batch

    def get_batch(self):
        pass
