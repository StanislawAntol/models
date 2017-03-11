# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import re

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.framework.python.ops import variables as variables_lib

import cifar10
import squeezenet as net

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train_gpu',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('save_checkpoint_secs', 10*60,
                            """Number of seconds between checkpoint saving.""")
tf.app.flags.DEFINE_integer('save_summaries_steps', 500,
                            """Number of steps between summary saving. """
                            """Too frequent can slow things down significantly.""")

def tower_loss(scope,
               is_training=True, 
               weight_decay=0.0001, 
               batch_norm_decay=0.999):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # Get images and labels for CIFAR-10.
  with tf.variable_scope('input_processing'):
    images, labels = cifar10.distorted_inputs()

  # Build a Graph that computes the logits predictions from the
  # inference model.
  with tf.variable_scope(tf.get_variable_scope()):
    logits, end_points = net.inference24(images,
            num_classes=10,
            is_training=is_training,
            scope=scope,
            weight_decay=weight_decay,
            batch_norm_decay=batch_norm_decay)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # Assemble all of the losses for the current tower only.
  losses = tf.losses.get_losses(scope=scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % net.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)
  
  return total_loss

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def train():
  is_training = True
  weight_decay = net.WEIGHT_DECAY
  bn_decay = net.BATCHNORM_MOVING_AVERAGE_DECAY
  
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Variables that affect learning rate.
    num_batches_per_epoch = net.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * net.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(net.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    net.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Create an optimizer that performs gradient descent.
    #opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.AdamOptimizer(lr, epsilon=0.001)

    # Calculate the gradients for each model tower.
    tower_grads = []
    summaries = []

    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (net.TOWER_NAME, i)) as scope:
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            with arg_scope([variables_lib.model_variable, variables_lib.variable], device='/cpu:0'):
              loss = tower_loss(scope, 
                                is_training=is_training, 
                                weight_decay=weight_decay, 
                                batch_norm_decay=bn_decay)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            ## Retain the summaries from the final tower.
            if i == (FLAGS.num_gpus-1):
              summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
              ## Retain the Batch Normalization updates operations only from the
              ## final tower. Ideally, we should grab the updates from all towers
              ## but these stats accumulate extremely fast so we can ignore the
              ## other stats from the other towers without significant detriment.
              #batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    with tf.variable_scope('optimizer'):
      # We must calculate the mean of each gradient. Note that this is the
      # synchronization point across all towers.
      grads = average_gradients(tower_grads)

      # Apply gradients.
      apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    # Assemble all of the losses for all towers
    losses = tf.losses.get_losses()
    total_loss = tf.add_n(losses, name='total_loss')

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        net.MOVING_AVERAGE_DECAY, global_step)
    variables_to_average = (tf.trainable_variables() +
                            tf.moving_average_variables())
    variables_averages_op = variable_averages.apply(variables_to_average)
    
    #with tf.control_dependencies([apply_gradient_op, variables_averages_op].extend(batchnorm_updates)):
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')
      
    uninit = tf.report_uninitialized_variables()

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        unin = tf.train.SessionRunArgs(uninit)
        print("Uninitialized list: ", unin)

      def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs(total_loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        loss_value = run_values.results / FLAGS.num_gpus
        if self._step % 10 == 0:
          num_examples_per_step = FLAGS.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                              examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        save_checkpoint_secs=FLAGS.save_checkpoint_secs,
        save_summaries_steps=FLAGS.save_summaries_steps,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
              tf.train.NanTensorHook(total_loss),
                _LoggerHook()],
        config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
        while not mon_sess.should_stop():
          mon_sess.run(train_op)

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()