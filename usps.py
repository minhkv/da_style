import h5py
import numpy as np
from skimage import transform
from dataset import *

class USPSDataset(Dataset):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        with h5py.File("USPS_data/usps.h5", "r") as hf:
            train = hf.get('train')
            self.X_tr = train.get('data')[:]
            self.y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]
        
        print(len(self.X_tr))
        self.current_index = 0
        
        self.num_train_batch = len(self.X_tr) // batch_size
