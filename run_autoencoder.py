from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt

from autoencoder import *
from domain_adaptation import *
from utils import *
from usps import *
from mnist import *
from tqdm import tqdm

tf.reset_default_graph()
# mnist = input_data.read_data_sets("MNIST_data")

batch_size = 100  # Number of samples in each batch
epoch_num = 5     # Number of epochs to train the network
lr = 0.0001        # Learning rate

# usps_data = USPSDataset(batch_size=batch_size)
mnist_data = MNISTDataset(batch_size=batch_size)
mnist_autoencoder = Autoencoder(name="source")
# usps_autoencoder = Autoencoder(name="target")
# domain_adaptation = DomainAdaptation(mnist_autoencoder, usps_autoencoder)
# domain_adaptation.merge_all()

mnist_autoencoder.init_variable()
mnist_autoencoder.merge_all() 

for step in tqdm(range(2)):
    batch_img, batch_label = mnist_data.next_batch_train() 
    mnist_autoencoder.fit(batch_img, step)
mnist_autoencoder.save_model()

