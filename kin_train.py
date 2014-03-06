"train a regression MLP"

import numpy as np
import cPickle as pickle
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

train_file = 'data/train.csv'
output_model_file = 'model.pkl'

hidden_size = 100
epochs = 50

# load data

train = np.loadtxt( train_file, delimiter = ',' )
x_train = train[:,0:-1]
y_train = train[:,-1]
y_train = y_train.reshape( -1, 1 )

input_size = x_train.shape[1]
target_size = y_train.shape[1]

# prepare dataset

ds = SDS( input_size, target_size )
ds.setField( 'input', x_train )
ds.setField( 'target', y_train )

# init and train

net = buildNetwork( input_size, hidden_size, target_size, bias= True )
trainer = BackpropTrainer( net,ds )

print "training for {} epochs...".format( epochs )

#trainer.trainUntilConvergence( verbose = True, validationProportion = 0.1, maxEpochs = epochs )

for i in range( epochs ):
	mse = trainer.train()
	rmse = sqrt( mse )
	print "training RMSE:", rmse
	
pickle.dump( net, open( output_model_file, 'wb' ))






