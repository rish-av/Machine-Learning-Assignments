#please install following before you run the code
#pip install SimpSOM


import SimpSOM as som
import numpy as np

filename = 'Data/pendigits.tes'
raw_data = open(filename, 'rt')
data = np.loadtxt(raw_data, delimiter=",")
data_without_label = data[:,:16]
labels = data[:,16]

#30x30 SOM with periodic boundary conditions
net = som.somNet(30, 30, data_without_label, PBC=True)
net.train(0.01, 10000)
net.save('weights')
net.nodes_graph(colnum=0)
net.diff_graph()
net.project(data_without_label, labels=labels)
net.cluster(data_without_label, type='qthresh')