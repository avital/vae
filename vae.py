% matplotlib
inline

import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import numpy as np
import matplotlib.pyplot as plt

latent_dim = 10
nn_width = 256

class VAE:
    def __init__(self, depth, width, n_z, free_nats_per_z):
        # batch norm
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.bn = lambda x: slim.batch_norm(x, is_training=self.is_training)

        # placeholder tensor for inputs
        self.inputs = tf.placeholder(tf.float32, shape=[None, 784], name="inputs")

        # encoder network (list of layers)
        self.encoder = [inputs]
        for i in range(depth):
            self.encoder.append(slim.fully_connected(self.encoder[-1], width, normalizer_fn=self.bn, activation_fn=tf.nn.elu, scope='encoder_' + (i+1)))

        # latent variable distributions
        self.z_mean = slim.fully_connected(self.encoder[-1], n_z, activation_fn=None, scope="z_mean")
        self.z_logstddev = slim.fully_connected(self.encoder[-1], n_z, activation_fn=None, scope="z_logstddev")
        self.z_stddev = tf.exp(self.z_logstddev)

        # 'noise' is unit gaussian; z is predicted latent distribution via the reparameterization trick
        self.noise = tf.placeholder(tf.float32, shape=[None, latent_dim], name="noise")
        self.z = self.z_mean + self.z_stddev * self.noise

        # KL loss in each latent dimension
        self.ind_kl_loss = 1 / 2 * (tf.pow(z_mean, 2) + tf.pow(z_stddev, 2) - 2 * tf.log(z_stddev) - 1)
        self.ind_kl_loss_with_free_nats = tf.maximum(free_nats_per_z, self.ind_kl_loss)
        # Reduce into single KL loss assuming diagonal Gaussian
        self.kl_loss = tf.reduce_sum(self.ind_kl_loss_with_free_bits, axis=1) # axis=0 is batch, axis=1 is z dim

        # decoder network (list of layers)
        self.decoder = [z]
        for i in range(depth):
            self.decoder.append(slim.fully_connected(decoder[-1], width, normalizer_fn=self.bn, activation_fn=tf.nn.elu)
        output = slim.fully_connected(self.decoder[-1], 784, normalizer_fn=self.bn, activation_fn=tf.nn.sigmoid) * (1 - 2e-6) + 1e-6

        # reconstruction loss
        self.reconst_loss = -inputs * tf.log(output) - (1 - inputs) * tf.log(1 - output)
        self.reconst_loss = tf.reduce_mean(self.reconst_loss, axis=1)

        # full loss
        self.loss = self.reconst_loss + self.kl_loss


vae = VAE(depth=2, width=256, n_z=10, free_nats_per_z=0.1)

tf.Session().run(tf.global_variables_initializer())

optimizer = tf.train.AdamOptimizer(epsilon=1e-3)
train_op = slim.learning.create_train_op(vae.loss, optimizer))

for epoch in range(1000000):
    batch_size = 1000
    images, labels = mnist.train.next_batch(batch_size)
    sampled_noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
    _, cur_output, cur_kl_loss, cur_reconst_loss, cur_loss, cur_z_mean, cur_z_stddev, cur_noise, cur_enc_fc1, cur_enc_fc2 = sess.run(
        [train_op, output, kl_loss, reconst_loss, loss, z_mean, z_stddev, noise, enc_fc1, enc_fc2],
        feed_dict={inputs: images, noise: sampled_noise, is_training: True})
    if epoch % 300 == 0:
        print("[{0}] Data probability: {1}".format(epoch, np.exp(-cur_loss[-1])))
        print("mean Loss: {0}".format(np.mean(cur_loss)))
        print("Noise vector: {0}".format(cur_noise[-1]))
        print("mean kl loss:", np.mean(cur_kl_loss))
        print("mean reconst loss:", np.mean(cur_reconst_loss))
        print("fc1:", cur_enc_fc1[-1][0:10])
        #        print("fc2:", cur_enc_fc2[-1][0:10])
        print("fc3:", cur_enc_fc3[-1][0:10])
        print("z mean:", cur_z_mean[-1])
        print("z stddev:", cur_z_stddev[-1])
        plt.imshow(np.reshape(cur_output[-1], [28, 28]), cmap="gray")
        plt.show()

