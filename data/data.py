class Data:
    def __init__(self, meta_data):
        self.size = 0
        self.data = meta_data.train
        self.dim = [28,28,3]

    def sample(self, quantity):
        pass

    def get_exemple(self):
        pass

    def get_next_batch(self, size):
        batch = self.data.next_batch(size)
        return batch

    def get_batch(self):
        pass
