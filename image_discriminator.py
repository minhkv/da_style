import tensorflow as tf
import numpy as np
from discriminator import *

lays = tf.layers

class ImageDiscriminator(Discriminator):
    def model(self, inputs):
        
        with tf.variable_scope("feature_extract_{}".format(self.name), reuse=tf.AUTO_REUSE):
            net = lays.conv2d(inputs, 64, [5, 5], strides=1, padding='SAME', activation=tf.nn.relu, name="C1")
            net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="S1")
            
            net = lays.conv2d(net, 128, [5, 5], strides=1, padding='VALID', activation=tf.nn.relu, name="C2")
            net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="S2")
            
            net = lays.conv2d(net, 256, [5, 5], strides=1, padding='VALID', activation=tf.nn.relu, name="C3")
            # net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="S3")
            net = lays.flatten(net, name="C3_flat")

        with tf.variable_scope("fully_connected_{}".format(self.name), reuse=tf.AUTO_REUSE):
            net = lays.dense(net, 256, activation=tf.nn.relu, name="F1")
            net = lays.dense(net, 128, activation=tf.nn.relu, name="F2")
            pred_class = lays.dense(net, 10, activation=tf.nn.relu, name="output")
            pred_class = tf.nn.softmax(pred_class, name="prob_class")
            pred_type = lays.dense(net, 1, activation=tf.nn.relu, name="type")
        return pred_class, pred_type
        
    def _construct_graph(self):
        
        self.inputs_real = self.endpoints["inputs_real"]
        self.inputs_fake = self.endpoints["inputs_fake"]
        self.vars_generator = self.endpoints["vars_generator"]
        self.class_labels = self.endpoints["class_labels"]
        
        with tf.variable_scope("discriminator_{}".format(self.name)) as scope:
            self.logits_real, self.type_pred_real = self.model(self.inputs_real)
        with tf.variable_scope(scope, reuse=True) as scope2:
            with tf.name_scope(scope2.original_name_scope):
                self.logits_fake, self.type_pred_fake = self.model(self.inputs_fake)
        
    def _construct_loss(self):
        
        with tf.variable_scope("loss_image_discriminator_{}".format(self.name)):
            loss_d_real = binary_cross_entropy(tf.ones_like(self.type_pred_real), self.type_pred_real)
            loss_d_fake = binary_cross_entropy(tf.zeros_like(self.type_pred_fake), self.type_pred_fake)
            loss_g_feature_real = tf.reduce_mean(binary_cross_entropy(tf.zeros_like(self.type_pred_real), self.type_pred_real)) # gen fake common
            
            self.loss_g_feature = tf.reduce_mean(loss_g_feature_real)
            self.loss_d_feature = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))
            
            # Class loss: only for real 
            self.class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_real, labels=self.class_labels))
            # total
            self.total_loss_g = self.loss_g_feature + self.class_loss
            self.total_loss_d = self.loss_d_feature + self.class_loss
        
        self.vars_g = self.vars_generator
        self.vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator_{}'.format(self.name))]
        # with tf.variable_scope("optimize_image_discriminator_{}".format(self.name)):
        #     self.optimizer_g_feature = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(self.loss_g_feature, var_list=vars_g)
        #     self.optimizer_d_feature = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(self.loss_d_feature, var_list=vars_d)
        #     self.optimizer_class = tf.train.AdamOptimizer(learning_rate=self.lr, name="optimize_class_discriminator_{}".format(self.name)).minimize(self.class_loss)
        
    
