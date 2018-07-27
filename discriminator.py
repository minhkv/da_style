import tensorflow as tf
import numpy as np

lays = tf.layers

def binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))

class Discriminator:
    def __init__(self, name="", endpoints={}, lr = 0.0001):
        self.name = name
        self.lr = lr
        self.endpoints = endpoints
        self._construct_graph()
        self._construct_loss()
        self._construct_summary()

    def model(self, inputs):
        pass
    
    def _construct_graph(self):
        pass
        
    def _construct_loss(self):
        pass
        
    def _construct_summary(self):
        tf.summary.scalar("type_loss_g_{}".format(self.name), self.loss_g_feature)
        tf.summary.scalar("type_loss_d_{}".format(self.name), self.loss_d_feature)
        tf.summary.scalar("class_loss_d_{}".format(self.name), self.class_loss)

    def init_variable(self):
        # initialize the network
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def merge_all(self):
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('/tmp/log', self.sess.graph)
        
    