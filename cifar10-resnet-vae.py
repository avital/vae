import tensorflow as tf

EXP_NAME = 'cifar10-wide-resnet-vae-z-2000-free-nats-0.25'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', 'cifar10/data_batch*',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', './train_dir/{0}'.format(EXP_NAME),
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', './logs/{0}'.format(EXP_NAME),
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')


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

tf.reset_default_graph()

from tensorflow.python.training import moving_averages


# TODO: remove l2 decay from generator

HParams = namedtuple('HParams',
                     'batch_size, num_classes, '
                     'num_residual_units, weight_decay_rate, '
                     'relu_leakiness, n_z')


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
    self.noise = tf.random_normal(shape=[hps.batch_size, hps.n_z], mean=0, stddev=1)
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.contrib.framework.get_or_create_global_step()
    with tf.variable_scope('encoder'):
      self._build_encoder()
    with tf.variable_scope('decoder'):
#      self._build_decoder_mlp()
      self._build_decoder()
    with tf.variable_scope('cost'):
      self._build_cost()
    
#    with tf.variable_scope('simple'):
#      self._simple_reconst()
    
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.summary.merge_all()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_encoder(self):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
      x = self._images
      x = x - 0.5
      print("encoder first shape: ", x.get_shape())
      x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    filters = [16, 160, 320, 640]

    with tf.variable_scope('unit_1_0'):
      x = self._residual(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        x = self._residual(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = self._residual(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_2_%d' % i):
        x = self._residual(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = self._residual(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_3_%d' % i):
        x = self._residual(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, self.hps.relu_leakiness)
    
    print("last encoder x shape", x.get_shape())
    
    with tf.variable_scope('latent'):
#      self.z_mean = 0.1 * self._conv('z_mean', x, 1, filters[3], self.hps.z_shape[2], [1, 1, 1, 1])
#      self.z_logstd = 0.1 * self._conv('z_logstd', x, 1, filters[3], self.hps.z_shape[2], [1, 1, 1, 1])
      with tf.variable_scope("z_mean"):
        self.z_mean = 0.1 * self._fully_connected(x, self.hps.n_z)
      with tf.variable_scope("z_logstd"):
        self.z_logstd = 0.1 * self._fully_connected(x, self.hps.n_z)
      self.z_logstd = tf.clip_by_value(self.z_logstd, -6, 6)
      self.z_std = tf.exp(self.z_logstd, name="z_std")
      self.z = self.noise * self.z_std + self.z_mean

  def _build_decoder_mlp(self):
    x = self.z
    with tf.variable_scope("init_fc_to_correct_dim"):
      x = self._fully_connected(x, 32 * 32 * 3)
      x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope("mlp1"):
      x = self._fully_connected(x, 32 * 32 * 3)
      x = self._relu(x, self.hps.relu_leakiness)
        
    with tf.variable_scope("mlp2"):
      x = self._fully_connected(x, 32 * 32 * 3)
      x = self._relu(x, self.hps.relu_leakiness)
        
    with tf.variable_scope("mlp3"):
      x = self._fully_connected(x, 32 * 32 * 3)
#      x = self._relu(x, self.hps.relu_leakiness)
        
    x = tf.reshape(x, [-1, 32, 32, 3])
    x = tf.sigmoid(x)
    self.reconstructed_image = x
    tf.summary.image("reconstructed", x)

    x = tf.reshape(x, [-1, 32, 32, 3])
    return x

  def _build_decoder(self):
    strides = [2, 2, 1]
    filters = [640, 320, 160, 16]
    activate_before_residual = [True, False, False]

    with tf.variable_scope('init'):
      x = self.z
      with tf.variable_scope("init_fc_to_correct_dim"):
        x = self._fully_connected(x, 8 * 8 * 640)
      x = tf.reshape(x, [-1, 8, 8, 640])
      x = self._conv('init_conv', x, 3, 640, 640, self._stride_arr(1))

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
    x = tf.sigmoid(x*0.1)
    self.reconstructed_image = x
    tf.summary.image("reconstructed", x)
        
    print("decoder last shape: ", x.get_shape())
      
  def _build_cost(self):
    with tf.variable_scope('costs'):
      # TODO: better loss function
      self.reconst_loss = -self._images * tf.log(self.reconstructed_image + 1e-4) - (1-self._images) * tf.log(1-self.reconstructed_image + 1e-4)
      self.reconst_loss = tf.reduce_sum(self.reconst_loss, axis=[1, 2, 3])
      self.reconst_loss = tf.reduce_mean(self.reconst_loss)

      self.ind_kl_loss = 1/2 * (tf.square(self.z_mean) + tf.square(self.z_std) - 2 * self.z_logstd - 1)
      self.ind_kl_loss = tf.maximum(0.25, self.ind_kl_loss) # free nats
      self.kl_loss = tf.reduce_sum(self.ind_kl_loss, axis=1) # axis=0 is batch, axis=1 is z dim
      self.kl_loss = tf.reduce_mean(self.kl_loss)
#      self.kl_loss = tf.constant(0, tf.float32)
    
      # TODO: Same decay?
      self.l2_loss = tf.constant(0, tf.float32)

      self.cost = self.reconst_loss + self.kl_loss
    
      tf.summary.scalar('reconst_loss', self.reconst_loss)
      tf.summary.scalar('kl_loss', self.kl_loss)
      tf.summary.scalar('l2_loss', self.l2_loss)
      tf.summary.scalar('cost', self.cost)
        
  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(0, tf.float32)
    tf.summary.scalar('learning_rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)
#    grads = tf.gradients(self.simple_reconst_loss, trainable_variables)

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
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
#      return y
      return x

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
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
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
        orig_x = self._conv_subpixel("conv_linear_identity_replacement", orig_x, 1, in_filter, out_filter, stride)

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
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _conv_subpixel(self, name, x, filter_size, in_filters, out_filters, strides):
    """Subpixel Convolution."""
    
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters * np.prod(strides)
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters * np.prod(strides)],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      x = tf.nn.conv2d(x, kernel, self._stride_arr(1), padding='SAME')

      # reshape
      assert strides[0] == 1
      assert strides[3] == 1
      shape = x.get_shape().as_list()
      x = tf.reshape(x, [shape[0], shape[1], shape[2], shape[3] // np.prod(strides), strides[1], strides[2]])
      x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
      x = tf.reshape(x, [shape[0], shape[1] * strides[1], shape[2] * strides[2], shape[3] // np.prod(strides)])
      return x
    
#  def _simple_reconst(self):
    
#    x = self._images
#    x = self._conv('conv', x, 3, 3, 12, [1, 2, 2, 1])
#    x = self._relu(x, 0.1)
#    # 1x1 conv
#    x = self._conv('conv1_1', x, 1, 12, 12, [1, 1, 1, 1])
#    x = self._relu(x, 0.1)

#    x = self._conv_subpixel('conv_subpixel', x, 3, 12, 3, [1, 2, 2, 1])
#    x = self._relu(x, 0.1)
#    # 1x1 conv
#    x = self._conv('conv1_2', x, 1, 3, 3, [1, 1, 1, 1])
#    x = tf.sigmoid(x)
    
#    self.simple_reconst = x
#    self.simple_reconst_loss = -self._images * tf.log(x + 1e-4) - (1-self._images) * tf.log(1-x + 1e-4)
#    self.simple_reconst_loss = tf.reduce_sum(self.simple_reconst_loss, axis=[1, 2, 3])
#    self.simple_reconst_loss = tf.reduce_mean(self.simple_reconst_loss)
    
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

import cifar_input

hps = HParams(batch_size=6,
              num_classes=10,
              num_residual_units=4,
#              num_residual_units=2,
              weight_decay_rate=0.0002,
              relu_leakiness=0.1,
              n_z=2000)

tf.reset_default_graph()

images, labels = cifar_input.build_input(FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode)

model = ResNet(hps, images, labels, FLAGS.mode)
model.build_graph()
init_op = tf.global_variables_initializer()
check_op = tf.add_check_numerics_ops()



#import matplotlib.pyplot as plt
#%matplotlib inline

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

"""ResNet Train/Eval module.
"""
import time
import six
import sys

import numpy as np

def train():
  """Training loop."""
  param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

  summary_writer = tf.summary.FileWriter(FLAGS.log_root)

  logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': model.cost,
               'reconst_loss': model.reconst_loss,
               'kl_loss': model.kl_loss,
               'l2_loss': model.l2_loss},
      every_n_iter=5)

  class _LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def begin(self):
      self._lrn_rate = 1e-3

    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
          model.global_step,  # Asks for global step value.
          feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
      train_step = run_values.results
    
      #if train_step < 40000:
      #  self._lrn_rate = 1e-4
      #elif train_step < 60000:
      #  self._lrn_rate = 3e-4
      #elif train_step < 80000:
      #  self._lrn_rate = 3e-5
      #else:
      #  self._lrn_rate = 3e-6
  with tf.train.MonitoredTrainingSession(
#      checkpoint_dir=FLAGS.log_root,
      hooks=[logging_hook, _LearningRateSetterHook()],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=0,
      config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as mon_sess:

    mon_sess.run(init_op)

    step = 0
    while not mon_sess.should_stop():
      step = step + 1
#      _, input_images, z, reconstructed_images = mon_sess.run([model.train_op, model._images, model.z_mean, model.reconstructed_image])
#      input_images, reconstructed_images = mon_sess.run([model._images, model.reconstructed_image])
      if step % 10 == 1:
        _, __, summaries, input_images, reconstructed_images = mon_sess.run([check_op, model.train_op, model.summaries, model._images, model.reconstructed_image])
        summary_writer.add_summary(summaries, global_step=step)
        summary_writer.flush()
        # XXX why did I need batch_size=32 here?
#        synthesized_images = mon_sess.run(model.reconstructed_image, feed_dict={model.z: np.random.normal(size=[hps.batch_size] + hps.z_shape)})
#        plt.subplot(131)
#        plt.imshow(input_images[-1])
#        plt.axis('off')
#        plt.subplot(132)
#        plt.imshow(reconstructed_images[-1])
#        plt.axis('off')
#        plt.subplot(133)
#        plt.imshow(synthesized_images[-1])
#        plt.show()
#        print(np.histogram(z))
      else:
        mon_sess.run([check_op, model.train_op])


def main(_):
  with tf.device('/gpu:0'):
    train()

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.app.run()
