import tensorflow as tf
# import tensorflow.layers as lays
import numpy as np
import scipy
from feature_discriminator import *
from image_discriminator import *
lays = tf.layers
def entropy_function(pk):
    return -pk * tf.log(pk)

class DomainAdaptation:
    def __init__(self, source_autoencoder, target_autoencoder, lr = 0.01, name="domain_adaptation"):
        self.source_autoencoder = source_autoencoder
        self.target_autoencoder = target_autoencoder
        
        self.lr = lr
        self.name = name
        self._construct_graph()
        self._construct_loss()
        self._construct_optimizer()
        self._construct_summary()
        
    def feature_classifier(self, inputs):
        with tf.variable_scope("feature_classifier_{}".format(self.name), reuse=tf.AUTO_REUSE):
            net = lays.dense(inputs, 128, activation=tf.nn.relu)
            net = lays.dense(net, 128, activation=tf.nn.relu)
            net = lays.dense(net, 10, activation=tf.nn.relu)
        return net
    
    def _construct_graph(self):

        # Exchange feature
        with tf.variable_scope("feature_exchange_to_source"):
            self.source_specific_latent = tf.placeholder(tf.float32, shape=(None, 1, 1, 128), name="source_specific_latent_{}".format(self.name))
            self.target_common_latent = tf.placeholder(tf.float32, shape=(None, 1, 1, 128), name="target_common_latent_{}".format(self.name))
            spe_source_com_target = tf.concat([self.source_specific_latent, self.target_common_latent], 3)
        with tf.variable_scope("feature_exchange_to_target"):
            self.target_specific_latent = tf.placeholder(tf.float32, shape=(None, 1, 1, 128), name="target_specific_latent_{}".format(self.name))
            self.source_common_latent = tf.placeholder(tf.float32, shape=(None, 1, 1, 128), name="source_common_latent_{}".format(self.name))
            spe_target_com_source = tf.concat([self.target_specific_latent, self.source_common_latent], 3)
        with tf.variable_scope("source_label"):
            self.source_label = tf.placeholder(tf.float32, (None, 10), name="source_label_{}".format(self.name)) 
        
        # Autoencoder varlist
        self.vars_encoder_source = [var for var in tf.trainable_variables() if var.name.startswith('encoder_{}'.format(self.source_autoencoder.name))]
        self.vars_encoder_target = [var for var in tf.trainable_variables() if var.name.startswith('encoder_{}'.format(self.target_autoencoder.name))]
        self.vars_decoder_source = [var for var in tf.trainable_variables() if var.name.startswith('decoder_{}'.format(self.source_autoencoder.name))]
        self.vars_decoder_target = [var for var in tf.trainable_variables() if var.name.startswith('decoder_{}'.format(self.target_autoencoder.name))]
        self.vars_feature_classifier = [var for var in tf.trainable_variables() if var.name.startswith('feature_classifier_{}'.format(self.name))]

        # Feed target input to source ae
        with tf.variable_scope(self.source_autoencoder.encoder_scope, reuse=True) as scope:
            with tf.name_scope(scope.original_name_scope):
                print("[Info] Encode target data by encoder_{}".format(self.source_autoencoder.name))
                self.latent_source_ae_target_data = self.source_autoencoder.encoder(self.target_autoencoder.ae_inputs)
                
        with tf.variable_scope(self.source_autoencoder.decoder_scope, reuse=True) as scope:
            with tf.name_scope(scope.original_name_scope):
                print("[Info] Decode target data by decoder_{}".format(self.source_autoencoder.name))
                self.reconstruct_source_target_data = self.source_autoencoder.decoder(self.latent_source_ae_target_data)

        with tf.variable_scope(self.source_autoencoder.decoder_scope, reuse=True) as scope:
            with tf.name_scope(scope.original_name_scope):
                print("[Info] Decode exchanged feature by decoder_{}".format(self.source_autoencoder.name))
                self.img_spe_source_com_target = self.source_autoencoder.decoder(spe_source_com_target)
                
        with tf.variable_scope(self.target_autoencoder.decoder_scope, reuse=True) as scope:
            with tf.name_scope(scope.original_name_scope):
                print("[Info] Decode exchanged feature by decoder_{}".format(self.target_autoencoder.name))
                self.img_spe_target_com_source = self.target_autoencoder.decoder(spe_target_com_source)
        
        # Feature classifier
        self.predict_source_common = self.feature_classifier(self.source_autoencoder.common)
        
        # Feature discriminator
        self.feature_discriminator = FeatureDiscriminator(
            name="df", 
            endpoints={
                "inputs_source": self.source_autoencoder.common,
                "inputs_target": self.target_autoencoder.common,
                "vars_generator_source": self.vars_encoder_source,
                "vars_generator_target": self.vars_encoder_target,
                "class_labels": self.source_label
            }
        )
        # Image discriminator
        self.image_discriminator_source = ImageDiscriminator(
            name="ds",
            endpoints={
                "inputs_real": self.source_autoencoder.ae_outputs,
                "inputs_fake": self.img_spe_source_com_target,
                "vars_generator": self.vars_decoder_source,
                "class_labels": self.source_label
            }
        )
        
        self.image_discriminator_target = ImageDiscriminator(
            name="dt",
            endpoints={
                "inputs_real": self.target_autoencoder.ae_outputs,
                "inputs_fake": self.img_spe_target_com_source,
                "vars_generator": self.vars_decoder_target,
                "class_labels": self.source_label
            }
        )
        
    def _construct_loss(self):
        # Construct feedback
        self.feedback_loss_source, self.feedback_loss_style_source = self._construct_feedback_loss(
            self.img_spe_source_com_target, 
            self.source_specific_latent, 
            self.target_common_latent,
            self.source_autoencoder)
            
        self.feedback_loss_target, self.feedback_loss_style_target = self._construct_feedback_loss(
            self.img_spe_target_com_source, 
            self.target_specific_latent, 
            self.source_common_latent,
            self.target_autoencoder)

        with tf.variable_scope("loss_autoencoder_{}".format(self.source_autoencoder.name)) as scope:
            with tf.name_scope(scope.original_name_scope):
                self.loss_reconstruct_source_img_target = tf.losses.mean_squared_error(self.target_autoencoder.ae_inputs, self.reconstruct_source_target_data)
        
        with tf.variable_scope("loss_feature_classifier"):
            # Feature classification loss
            self.loss_feature_classifier = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predict_source_common, labels=self.source_label), name="loss_feature_classifier")
        
        with tf.variable_scope("loss_entropy"):
            print(self.image_discriminator_source.logits_fake)
            loss_entropy_gs = entropy_function(self.image_discriminator_source.logits_fake)
            loss_entropy_xt = entropy_function(self.image_discriminator_target.logits_real)
            loss_entropy_ct = entropy_function(self.feature_discriminator.logits_source)
            self.loss_entropy = loss_entropy_gs + loss_entropy_xt + loss_entropy_ct
        with tf.variable_scope("loss_semantic"):
            ds_gs = self.image_discriminator_source.logits_fake
            dt_xt = self.image_discriminator_target.logits_real
            self.loss_semantic = tf.losses.mean_squared_error(ds_gs, dt_xt)
        
        with tf.name_scope("Step1"):
            self.loss_step1 = self.loss_feature_classifier
            var_step1 = self.vars_encoder_source + self.vars_feature_classifier
            self.optimizer_step1 = tf.train.GradientDescentOptimizer(learning_rate=0.01, name="optimize_1").minimize(self.loss_step1, var_list=var_step1)
        with tf.name_scope("Step2"):
            self.loss_step2 = self.loss_feature_classifier + self.source_autoencoder.loss + self.loss_reconstruct_source_img_target
            var_step2 = self.vars_feature_classifier + self.vars_encoder_source + self.vars_decoder_source
            self.optimizer_step2 = tf.train.GradientDescentOptimizer(learning_rate=0.001, name="optimize_2").minimize(self.loss_step2, var_list=var_step2)
            
            # self.loss_step2_usps = self.source_autoencoder.loss
            # var_step2_usps = self.vars_encoder_source + self.vars_decoder_source
            # self.optimizer_step2_usps = tf.train.GradientDescentOptimizer(learning_rate=0.01, name="optimize_2_usps").minimize(self.loss_step2_usps, var_list=var_step2_usps)
            
        with tf.name_scope("Step3"):
            self.loss_step3_g = 10 * self.loss_feature_classifier + self.source_autoencoder.loss + self.target_autoencoder.loss + self.feature_discriminator.loss_g_feature
            self.loss_step3_d = self.feature_discriminator.loss_d_feature
            varlist_g = self.vars_feature_classifier + self.vars_encoder_source + self.vars_encoder_target + self.vars_decoder_source + self.vars_decoder_target
            varlist_d = self.feature_discriminator.vars_d
            self.optimizer_step3_g = tf.train.GradientDescentOptimizer(learning_rate=self.lr * 0.1, name="optimize_3_g").minimize(self.loss_step3_g, var_list=varlist_g)
            self.optimizer_step3_d = tf.train.GradientDescentOptimizer(learning_rate=self.lr * 0.1, name="optimize_3_d").minimize(self.loss_step3_d, var_list=varlist_d)
        with tf.name_scope("Step4"):
            self.loss_step4 = 10 * self.loss_feature_classifier + self.source_autoencoder.loss + self.target_autoencoder.loss + self.feature_discriminator.total_loss_g
            self.optimizer_step4 = tf.train.GradientDescentOptimizer(learning_rate=self.lr, name="semantic_optimize").minimize(self.loss_step4)
        
    def _construct_optimizer(self):
        pass
        # with tf.variable_scope("optimize_feature_classifier"):
        #     self.optimize_feature_classifier = tf.train.GradientDescentOptimizer(learning_rate=self.lr, name="feature_classifier_optimize").minimize(self.loss_feature_classifier)
        
        
    def _construct_feedback_loss(self, gen_img, spe_latent, com_latent, autoencoder):
        print("[Info] Construct feedback {}".format(autoencoder.name))
        with tf.variable_scope(autoencoder.encoder_scope, reuse=tf.AUTO_REUSE) as scope:
            with tf.name_scope(scope.original_name_scope):
                feature_feedback = autoencoder.encoder(gen_img)
                spe, com = tf.split(feature_feedback, num_or_size_splits=2, axis=3)
        with tf.variable_scope("loss_L2_{}".format(autoencoder.name)):
            loss_spe = tf.losses.mean_squared_error(spe_latent, spe)
            loss_com = tf.losses.mean_squared_error(com_latent, com)
            loss_fea = loss_spe + loss_com
                
        with tf.variable_scope(autoencoder.decoder_scope, reuse=tf.AUTO_REUSE) as scope:
            with tf.name_scope(scope.original_name_scope):
                rec_feed_img = autoencoder.decoder(feature_feedback)
                
        with tf.variable_scope("loss_rec_feed_{}".format(self.name)):
            loss_rec_feed = tf.losses.mean_squared_error(gen_img, rec_feed_img, scope="loss_{}".format(self.name))
            
        return loss_fea, loss_rec_feed
        
    def _construct_summary(self):
        tf.summary.image("spe_source_com_target", self.img_spe_source_com_target, 3)
        tf.summary.image("spe_target_com_source", self.img_spe_target_com_source, 3)
        tf.summary.image("reconstruct_target_data", self.reconstruct_source_target_data, 3)
        tf.summary.scalar("feature_classifier_loss", self.loss_feature_classifier)
        tf.summary.scalar("source_reconstruct_target_data", self.loss_reconstruct_source_img_target)
        tf.summary.scalar("feedback_loss_source", self.feedback_loss_source)
        tf.summary.scalar("feedback_loss_target", self.feedback_loss_target)
        
    def merge_all(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('/tmp/log', self.sess.graph)
        
    def _feed_dict(self, batch_source, batch_target, source_label=[]):
        spe_source, com_source = self.source_autoencoder.get_split_feature(batch_source, self.sess)
        spe_target, com_target = self.target_autoencoder.get_split_feature(batch_target, self.sess)
        s_label = np.ones((batch_source.shape[0], 10))
        
        if source_label != []:
            s_label = self.sess.run(tf.one_hot(source_label, depth=10))
        return {
            self.source_specific_latent: spe_source, 
            self.source_common_latent: com_source,
            self.target_specific_latent: spe_target, 
            self.target_common_latent: com_target, 
            self.source_autoencoder.ae_inputs: batch_source, 
            self.target_autoencoder.ae_inputs: batch_target,
            self.source_label: s_label
        }
    
    def run_step1(self, batch_source, batch_target, source_label, step):
        summary, loss, _ = self.sess.run(
            [self.merged, self.loss_step1, self.optimizer_step1],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label)
        )
        print("Iter {}: loss step1: {:.4f}".format(step, loss))
        self.train_writer.add_summary(summary, step)
        
    def run_step2(self, batch_source, batch_target, source_label, step):
        summary, loss_1, _ = self.sess.run(
            [self.merged, self.loss_step2, self.optimizer_step2],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label)
        )
        print("Iter {}: loss step2: {:.4f}".format(step, loss_1))
        self.train_writer.add_summary(summary, step)
    
    def run_step3(self, batch_source, batch_target, source_label, step):
        summary, loss_g, _, loss_d, _ = self.sess.run(
            [self.merged, self.loss_step3_g, self.optimizer_step3_g, self.loss_step3_d, self.optimizer_step3_d],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label)
        )
        print("Iter {}: loss step3 g: {:.4f}, loss step3 d: {:.4f}".format(step, loss_g, loss_d))
        self.train_writer.add_summary(summary, step)
    
    # Feature classifier
    def run_optimize_feature_classifier(self, batch_source, batch_target, source_label, step):
        summary, loss_fc, _ = self.sess.run(
            [self.merged, self.loss_feature_classifier, self.optimize_feature_classifier],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label)
        )
        print("Iter {}: loss feature classifier: {:.4f}".format(step, loss_fc))
        self.train_writer.add_summary(summary, step)    
    