import SimpSOM as som
import numpy as np
from .. Dataset import active_learning

filename = 'Data/pendigits.tra'
raw_data = open(filename, 'rt')
data = np.loadtxt(raw_data, delimiter=",")
data_without_label = data[:100,:16]
labels = data[:100,16]

#30x30 SOM with periodic boundary conditions
net = som.somNet(30, 30, data_without_label, PBC=True)
net.train(0.01, 1000)
net.save('weights')
net.nodes_graph(colnum=0)
net.diff_graph()
net.project(raw_data, labels=labels)
net.cluster(raw_data, type='qthresh')	