from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt

from autoencoder import *
from discriminator import *
from mnist import *
import tensorflow as tf


# restore_source = Autoencoder("source", 
# meta_graph="/tmp/model/ae_source.meta", 
# checkpoint_dir="/tmp/model/ae_source")
# tf.reset_default_graph()
# sess = tf.Session()

batch_size = 100  # Number of samples in each batch

mnist_data = MNISTDataset(batch_size=batch_size)

discriminator = Discriminator(name="source")
discriminator.merge_all()
discriminator.init_variable()
# sess.run(tf.global_variables_initializer())

for i in range(200):
    batch_img, batch_label = mnist_data.next_batch_train() 
#     batch_label = sess.run(tf.one_hot(batch_label, depth=10))
    discriminator.fit(batch_img, batch_label, i)


# restore_source.merge_all()
# restore_source.fit(batch_img, 1)

# output_img = restore_source.forward(batch_img)
# plt.imshow(np.reshape(output_img[0], (32, 32)), cmap='gray')
# plt.show()
