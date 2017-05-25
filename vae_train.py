import tensorflow as tf

import vae_model

EXP_NAME = vae_model.MODEL_NAME + '-4batch-2000z-1'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar100', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', 'cifar100/train*',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', './logs/{0}/train'.format(EXP_NAME),
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


from tensorflow.python.training import moving_averages

HParams = namedtuple('HParams',
                     'batch_size, num_classes, '
                     'num_residual_units, weight_decay_rate, '
                     'relu_leakiness, n_z')


import vae_model

import cifar_input

hps = HParams(batch_size=4,
              num_classes=10,
              num_residual_units=4,
#              num_residual_units=2,
              weight_decay_rate=0.0002,
              relu_leakiness=0.1,
              n_z=2000)




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
  images, labels = cifar_input.build_input(FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode)

  model = vae_model.ResNet(hps, images, labels, FLAGS.mode)
  model.build_graph()
  init_op = tf.global_variables_initializer()
  check_op = tf.add_check_numerics_ops()

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
    
      if train_step < 20000:
        self._lrn_rate = 1e-4
      elif train_step < 40000:
        self._lrn_rate = 1e-5
      elif train_step < 60000:
        self._lrn_rate = 1e-6
      else:
        self._lrn_rate = 1e-7
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.log_root,
      hooks=[logging_hook, _LearningRateSetterHook()],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=0,
      is_chief=True,
      save_checkpoint_secs=60 * 10,
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
