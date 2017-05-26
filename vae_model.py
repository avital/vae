import tensorflow as tf

MODEL_NAME = 'cifar-wide-resnet-1-logistic-decoder-3'

#tf.app.flags.DEFINE_string('train_dir', './train_dir/{0}'.format(EXP_NAME),
#                           'Directory to keep training outputs.')
#tf.app.flags.DEFINE_string('log_root', './logs/{0}'.format(EXP_NAME),
#                           'Directory to keep the checkpoints. Should be a '
#                           'parent directory of FLAGS.train_dir/eval_dir.')

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages

class ResNet(object):
    """ResNet model."""

    def __init__(self, hps, images, labels, mode):
        """ResNet constructor.

        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]
          mode: One of 'train' and 'eval'.
        """
        self.hps = hps
        self._images = images
        self.noise = tf.random_normal(shape=[hps.batch_size, 8, 8, 640], mean=0, stddev=1)
        self.labels = labels
        self.mode = mode

        self._extra_train_ops = []

    def build_graph(self):
        """Build a whole graph for the model."""
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.summaries = []
        self.reconst_summaries = []

        with tf.variable_scope('logistic'):
            self.logistic_logs = tf.get_variable("logistic_logs", initializer=tf.constant(np.log(10/255.), dtype=tf.float32))
            self.logistic_s = tf.exp(tf.clip_by_value(self.logistic_logs, -6, 6))
            self.summaries.append(tf.summary.scalar('logistic_s', self.logistic_s))

        with tf.variable_scope('encoder'):
            self._build_encoder()
        with tf.variable_scope('decoder'):
            self._build_decoder()
        with tf.variable_scope('cost'):
            self._build_cost()

        if self.mode == 'train':
            self._build_train_op()
        self.summaries_merged = tf.summary.merge(self.summaries + self.reconst_summaries)
        self.summaries_merged_sampled = tf.summary.merge(self.summaries + [self.sampled_summary])

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_encoder(self):
        """Build the core model within the graph."""
        with tf.variable_scope('init'):
            self.reconst_summaries.append(tf.summary.image('images', self._images))
            x = self._images - 0.5
            print("encoder first shape: ", x.get_shape())
            x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

        self.encoder0 = x

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        filters = [16, 160, 320, 640]

        with tf.variable_scope('unit_1_0'):
            x = self._residual(x, filters[0], filters[1], self._stride_arr(strides[0]),
                               activate_before_residual[0])
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_1_%d' % i):
                x = self._residual(x, filters[1], filters[1], self._stride_arr(1), False)

        self.encoder1 = x

        with tf.variable_scope('unit_2_0'):
            x = self._residual(x, filters[1], filters[2], self._stride_arr(strides[1]),
                               activate_before_residual[1])
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_2_%d' % i):
                x = self._residual(x, filters[2], filters[2], self._stride_arr(1), False)

        self.encoder2 = x

        with tf.variable_scope('unit_3_0'):
            x = self._residual(x, filters[2], filters[3], self._stride_arr(strides[2]),
                               activate_before_residual[2])
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_3_%d' % i):
                x = self._residual(x, filters[3], filters[3], self._stride_arr(1), False)

        self.encoder3 = x

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)

        print("last encoder x shape", x.get_shape())

        with tf.variable_scope('latent'):
            self.z_mean = 0.1 * self._conv('z_mean', x, 1, filters[3], filters[3], [1, 1, 1, 1])
            self.z_logstd = 0.1 * self._conv('z_logstd', x, 1, filters[3], filters[3], [1, 1, 1, 1])
            self.z_logstd = tf.clip_by_value(self.z_logstd, -6, 6)
            self.z_std = tf.exp(self.z_logstd, name="z_std")
            self.z = self.noise * self.z_std + self.z_mean

    def _build_decoder(self):
        strides = [2, 2, 1]
        filters = [640, 320, 160, 16]
        activate_before_residual = [True, False, False]

        with tf.variable_scope('init'):
            x = self.z
            #with tf.variable_scope("init_conv"):
#                x = self._fully_connected(x, 8 * 8 * filters[0])
#            x = tf.reshape(x, [-1, 8, 8, filters[0]])
            x = self._conv('init_conv', x, 3, filters[0], filters[0], self._stride_arr(1))

        with tf.variable_scope('unit_1_0'):
            x = self._backresidual(x, filters[0], filters[1], self._stride_arr(strides[0]),
                                   activate_before_residual[0])
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_1_%d' % i):
                x = self._backresidual(x, filters[1], filters[1], self._stride_arr(1), False)

        with tf.variable_scope('unit_2_0'):
            x = self._backresidual(x, filters[1], filters[2], self._stride_arr(strides[1]),
                                   activate_before_residual[1])
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_2_%d' % i):
                x = self._backresidual(x, filters[2], filters[2], self._stride_arr(1), False)

        with tf.variable_scope('unit_3_0'):
            x = self._backresidual(x, filters[2], filters[3], self._stride_arr(strides[2]),
                                   activate_before_residual[2])
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_3_%d' % i):
                x = self._backresidual(x, filters[3], filters[3], self._stride_arr(1), False)

        # to RGB image
        x = self._conv('to_image', x, 1, filters[3], 3, [1, 1, 1, 1])
        x = tf.sigmoid(x * 0.1)
        self.reconstructed_image = x

        self.reconst_summaries.append(tf.summary.image("reconstructed", x))
        self.sampled_summary = tf.summary.image("sampled", x)

        print("decoder last shape: ", x.get_shape())

    def _build_cost(self):
        with tf.variable_scope('costs'):
            image_low_value = tf.floor(self._images * 254.9999) / 255.
            image_high_value = image_low_value + 1 / 255.
            def cdf(x, mu):
                return tf.sigmoid((x - mu) / self.logistic_s)
            low_cdf = cdf(image_low_value, self.reconstructed_image)
            high_cdf = cdf(image_high_value, self.reconstructed_image)
            self.logpx_z = -tf.log(1e-6 + high_cdf - low_cdf)
            self.reconst_loss = tf.reduce_sum(self.logpx_z, axis=[1,2,3])

            self.ind_kl_loss = 1 / 2 * (tf.square(self.z_mean) + tf.square(self.z_std) - 2 * self.z_logstd - 1)
            self.base_kl_loss = tf.reduce_sum(self.ind_kl_loss, axis=[1,2,3])  # axis=0 is batch, axis=1 is z dim
            self.kl_loss = tf.reduce_sum(tf.maximum(0.25, tf.reduce_mean(tf.reduce_sum(self.ind_kl_loss, axis=[1,2]), axis=0)))
#            self.ind_kl_loss = tf.maximum(0.25, self.ind_kl_loss)  # free nats
 #           self.kl_loss = tf.reduce_sum(self.ind_kl_loss, axis=[1,2,3])  # axis=0 is batch, axis=1 is z dim
#            self.kl_loss = tf.reduce_mean(self.kl_loss)

            self.cost = tf.reduce_mean(self.reconst_loss + self.kl_loss, axis=0)
            self.base_cost = self.reconst_loss + self.base_kl_loss


            self.reconst_summaries.append(tf.summary.scalar('reconst_loss', tf.reduce_mean(self.reconst_loss, 0)))
            self.summaries.append(tf.summary.scalar('kl_loss', self.kl_loss))
            self.summaries.append(tf.summary.scalar('base_kl_loss', tf.reduce_mean(self.base_kl_loss, 0)))
            self.reconst_summaries.append(tf.summary.scalar('cost', self.cost))
            self.reconst_summaries.append(tf.summary.scalar('base_cost', tf.reduce_mean(self.base_cost, 0)))

            self.est_mar_nll_bits_per_subpixel = tf.placeholder(tf.float32)
            self.summaries.append(tf.summary.scalar('est_marginal_nll_bits_per_subpixel', self.est_mar_nll_bits_per_subpixel))

    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.constant(0, tf.float32)
        self.summaries.append(tf.summary.scalar('learning_rate', self.lrn_rate))

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lrn_rate, epsilon=1e-3)

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

    # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)
                self.summaries.append(tf.summary.histogram(mean.op.name, mean))
                self.summaries.append(tf.summary.histogram(variance.op.name, variance))
            # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _backresidual(self, x, in_filter, out_filter, stride,
                      activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub2'):
            x = self._conv('conv2', x, 3, in_filter, in_filter, [1, 1, 1, 1])
            x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub1'):
            x = self._conv_subpixel('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = self._conv_subpixel("conv_linear_identity_replacement", orig_x, 1, in_filter, out_filter,
                                             stride)

            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.summary.histogram(var.op.name, var)

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _conv_subpixel(self, name, x, filter_size, in_filters, out_filters, strides):
        """Subpixel Convolution."""

        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters * np.prod(strides)
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters * np.prod(strides)],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            x = tf.nn.conv2d(x, kernel, self._stride_arr(1), padding='SAME')

            # reshape
            assert strides[0] == 1
            assert strides[3] == 1
            shape = x.get_shape().as_list()
            x = tf.reshape(x, [shape[0], shape[1], shape[2], shape[3] // np.prod(strides), strides[1], strides[2]])
            x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
            x = tf.reshape(x, [shape[0], shape[1] * strides[1], shape[2] * strides[2], shape[3] // np.prod(strides)])
            return x

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        x = tf.reshape(x, [self.hps.batch_size, -1])
        w = tf.get_variable(
            'DW', [x.get_shape()[1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

