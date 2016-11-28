import numpy as np
import math
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

class Controler:
    def __init__(self):
        self.last_epoch = 0
        self.n_epoch = 2000
        self.n_batch_epoch = 4
        self.batch_size = 10
        self.n_exemple_averaging = 10
        self.n_batch_averaging = math.floor(self.n_exemple_averaging / self.batch_size)
        self.n_batch= (self.n_epoch - self.last_epoch) * self.n_batch_epoch

        self.plot_ = {'train_loss':np.empty(1)}
        self.save_ = None

        stream_tokens = tls.get_credentials_file()['stream_ids']
        token_1 = stream_tokens[-1]
        self.loss_plot = py.Stream(stream_id=token_1)

        stream_id1 = dict(token=token_1, maxpoints=60)
        self.data_plot = go.Scatter(x=[], y=[], stream=stream_id1, name='trace1')
        fig = go.Figure(data=[self.data_plot])

        #self.loss_plot.data=[self.data_plot]
        self.loss_plot.open()

    def plot(self):
        y = [2, 5,6,7,4]
        self.loss_plot.write(dict(y=y))


        #self.loss_plot.data=[self.data_plot]
        #py.plot(self.loss_plot)
