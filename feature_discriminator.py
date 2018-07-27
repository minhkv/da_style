import tensorflow as tf
import numpy as np
from discriminator import *

lays = tf.layers

class FeatureDiscriminator(Discriminator):
    def model(self, inputs):
        net = lays.dense(inputs, 128, activation=tf.nn.relu, name="F1")
        net = lays.dense(net, 128, activation=tf.nn.relu, name="F2")
        pred_class = lays.dense(net, 10, activation=tf.nn.relu, name="output_class")
        pred_class = tf.nn.softmax(pred_class, name="prob_class")
        pred_type = lays.dense(net, 1, activation=tf.nn.relu, name="output_type")
        return pred_class, pred_type
    
    def _construct_graph(self):
        self.inputs_source = self.endpoints["inputs_source"]
        self.inputs_target = self.endpoints["inputs_target"]
        self.vars_generator_source = self.endpoints["vars_generator_source"]
        self.vars_generator_target = self.endpoints["vars_generator_target"]
        self.class_labels = self.endpoints["class_labels"]
        
        with tf.variable_scope("discriminator_{}".format(self.name)) as scope:
            self.logits_source, self.type_pred_source = self.model(self.inputs_source)
        
        with tf.variable_scope(scope, reuse=True) as scope2:
            with tf.name_scope(scope2.original_name_scope):
                self.logits_target, self.type_pred_target = self.model(self.inputs_target)

    def _construct_loss(self):
        # Type loss: source or target
        # Adversarial learning
        # source: type 1, target type: 0
        with tf.variable_scope("loss_feature_discriminator_{}".format(self.name)):
            loss_d_source = binary_cross_entropy(tf.ones_like(self.type_pred_source), self.type_pred_source)
            loss_d_target = binary_cross_entropy(tf.zeros_like(self.type_pred_target), self.type_pred_target)
            loss_g_feature_source = tf.reduce_mean(binary_cross_entropy(tf.zeros_like(self.type_pred_source), self.type_pred_source)) # gen target common
            loss_g_feature_target = tf.reduce_mean(binary_cross_entropy(tf.ones_like(self.type_pred_target), self.type_pred_target)) # gen source common
            self.loss_g_feature = tf.reduce_mean(0.5 * (loss_g_feature_source + loss_g_feature_target))
            self.loss_d_feature = tf.reduce_mean(0.5 * (loss_d_source + loss_d_target))
            
            # Class loss: only for source 
            self.class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_source, labels=self.class_labels))
            self.total_loss_g = self.loss_g_feature + self.class_loss
            self.total_loss_d = self.loss_d_feature + self.class_loss
        self.vars_g = self.vars_generator_source + self.vars_generator_target
        self.vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator_{}'.format(self.name))]
    
    def _construct_summary(self):
        tf.summary.scalar("type_loss_g_{}".format(self.name), self.loss_g_feature)
        tf.summary.scalar("type_loss_d_{}".format(self.name), self.loss_d_feature)
        tf.summary.scalar("class_loss_d_{}".format(self.name), self.class_loss)