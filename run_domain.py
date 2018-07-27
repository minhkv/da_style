from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt

from autoencoder import *
from feature_discriminator import *
from domain_adaptation import *
from utils import *
from usps import *
from mnist import *

tf.reset_default_graph()

batch_size = 100  # Number of samples in each batch

usps_data = USPSDataset(batch_size=batch_size)
mnist_data = MNISTDataset(batch_size=batch_size)
mnist_autoencoder = Autoencoder(name="source")
usps_autoencoder = Autoencoder(name="target")

domain_adaptation = DomainAdaptation(mnist_autoencoder, usps_autoencoder)
domain_adaptation.merge_all()

r_1_fc = 1000
r_2_rec = 1000
r_3_df = 1000
r_4_di = 10
current_step = 0

for step in (range(r_1_fc)):
    batch_img, batch_label = mnist_data.next_batch_train() 
    batch_target, label_target = usps_data.next_batch_train()
    domain_adaptation.run_step1(batch_img, batch_target, batch_label,  step + current_step)
current_step += r_1_fc

for step in (range(r_2_rec)):
    batch_img, batch_label = mnist_data.next_batch_train()
    batch_target, label_target = usps_data.next_batch_train()
    domain_adaptation.run_step2(batch_img, batch_target, batch_label,  step + current_step)
current_step += r_2_rec

for step in (range(r_3_df)):
    batch_img, batch_label = mnist_data.next_batch_train()
    batch_target, label_target = usps_data.next_batch_train()
    domain_adaptation.run_step3(batch_img, batch_target, batch_label,  step + current_step)
current_step += r_3_df

# for step in (range(r_4_di)):
#     batch_img, batch_label = mnist_data.next_batch_train()
#     batch_target, label_target = usps_data.next_batch_train()
#     domain_adaptation.run_step4(batch_img, batch_target, batch_label,  step + current_step)
