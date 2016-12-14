import tensorflow as tf
from data.data import Data
from data.metadata import get_metadata
from monitor import Monitor
from train import Train
from test import Test
from model import Model
from config import init_config


opts = init_config()
#initialization of datasets
meta_data = get_metadata()
train_set = Data(meta_data['train'])
#Val_set = Data(meta_data['val'])

#initialization of controle config
monitor = Monitor(opts)

#construction of the model
model = Model(opts)

#training
trainer = Train(monitor, model)
trainer.train(opts, monitor, train_set, model)

#testing
tester = Test(model)
tester.test(monitor, train_set, model)
