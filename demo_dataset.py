from mnist import *
from usps import *

mnist = MNISTDataset(100)
usps = USPSDataset(100)

for i in range(2000):
    mnist.next_batch_train()