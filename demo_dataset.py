from mnist import *
from usps import *

mnist = MNISTDataset(100)
usps = USPSDataset(100)
usps.sample_dataset(1800)
for i in range(2000):
    usps.next_batch_train()