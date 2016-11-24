import tensorflow as tf
from data.data import Data
from data.metadata import *
from controle import Controler
from train import Train
from model import Model


#initialization of train dataset
meta_data = get_metadata()
train_set = Data(meta_data.train)
controler = Controler()

model = Model()
trainer = Train(model)
trainer.train(controler, train_set, model)
