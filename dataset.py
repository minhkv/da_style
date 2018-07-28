import numpy as np
from skimage import transform


class Dataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.X_tr = []
        self.y_tr = []
        self._initialize()
        
        
    def _initialize(self):
        self.current_index = 0
        self.num_train_batch = len(self.X_tr) // self.batch_size
        self.dataset_size = len(self.X_tr)
    def _resize_batch(self, imgs):
        imgs = self._reshape_images(imgs)
        resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
        for i in range(imgs.shape[0]):
            resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
        return resized_imgs

    def _reshape_images(self, imgs):
        return imgs.reshape((-1, 16, 16, 1))

    def sample_dataset(self, num_sample):
        self.dataset_size = num_sample

    def next_batch_train(self):
        print(self.current_index)
        addition_from_start = 0
        start = self.current_index
        end = start + self.batch_size
        if end > self.dataset_size:
            addition_from_start = end - self.dataset_size
            end = self.dataset_size
            self.current_index = addition_from_start
        else: 
            self.current_index += self.batch_size
        
        data = self.X_tr[start:end]
        label = self.y_tr[start: end]
        if addition_from_start > 0:
            data = np.concatenate((data, self.X_tr[0:addition_from_start]), axis=0)
            label = np.concatenate((label, self.y_tr[0:addition_from_start]), axis=0)
        data = self._resize_batch(data)
        assert data.shape == (self.batch_size, 32, 32, 1)
        return data, label
        
