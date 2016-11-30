import tensorflow as tf
from data.data import Data
from data.metadata import *
from controle import Controler
from train import Train
from test import Test
from model import Model
from config import init_config


opts = init_config()
#initialization of train dataset
meta_data = get_metadata()
train_set = Data(meta_data['train'])
controler = Controler()

model = Model(opts)
trainer = Train(controler, model)
trainer.train(controler, train_set, model)

tester = Test(model)
tester.test(controler, train_set, model)
