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

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow import concat

from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.framework import add_arg_scope

from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.layers.python.layers import layers as layers
from tensorflow.contrib.layers.python.layers import regularizers


import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.

# If a model is trained using multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9

WEIGHT_DECAY = 0.00004

NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss, scope=None):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses', scope=scope)
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name + ' (raw)', l)
    tf.summary.scalar(loss_name, loss_averages.average(l))

  return loss_averages_op

def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

@add_arg_scope # Adds arguments so it can be used with arg_scope
def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                outputs_collections=None):
  with vs.variable_scope(scope, 'fire', [inputs], reuse=reuse) as sc:
    with arg_scope([layers.conv2d, layers.max_pool2d],
                   outputs_collections=None):
      net = squeeze(inputs, squeeze_depth)
      outputs = expand(net, expand_depth)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)

def squeeze(inputs, num_outputs):
    return layers.conv2d(inputs, num_outputs, [1, 1], stride=1, padding='VALID', scope='squeeze')

def expand(inputs, num_outputs):
  with vs.variable_scope('expand'):
    e1x1 = layers.conv2d(inputs, num_outputs, [1, 1], stride=1, padding='VALID', scope='1x1')
    e3x3 = layers.conv2d(inputs, num_outputs, [3, 3], stride=1, padding='SAME', scope='3x3')
    return concat([e1x1, e3x3], 3)

# Add dropout?
# Providing values in vs.variable_scope makes sure you're adding to/using the same graph
# For ImageNet
def inference227(inputs,
                 num_classes=1001,
                 is_training=True,
                 scope='squeezenet',
                 weight_decay=WEIGHT_DECAY,
                 batch_norm_decay=BATCHNORM_MOVING_AVERAGE_DECAY):
  with arg_scope(squeezenet_arg_scope(is_training, weight_decay, batch_norm_decay)):
    with vs.variable_scope('squeezenet', values=[inputs]) as sc:
      end_point_collection = sc.original_name_scope + '_end_points'
      with arg_scope([fire_module, layers.conv2d,
                      layers.max_pool2d, layers.avg_pool2d], 
                  outputs_collections=[end_point_collection]):
        net = layers.conv2d(inputs, 64, [3, 3], stride=2, padding='VALID', scope='conv01') # VALID?
        net = layers.max_pool2d(net, [3, 3], stride=2, scope='pool01')
        net = fire_module(net, 16, 64, scope='fire02')
        net = fire_module(net, 16, 64, scope='fire03')
        net = layers.max_pool2d(net, [3, 3], stride=2, scope='pool03')
        net = fire_module(net, 32, 128, scope='fire04')
        net = fire_module(net, 32, 128, scope='fire05')
        net = layers.max_pool2d(net, [3, 3], stride=2, scope='maxpool05')
        net = fire_module(net, 48, 192, scope='fire06')
        net = fire_module(net, 48, 192, scope='fire07')
        net = fire_module(net, 64, 256, scope='fire08')
        net = fire_module(net, 64, 256, scope='fire09')
        net = layers.conv2d(net, num_classes, [1, 1], 
                            stride=1, padding='VALID',
                            weights_initializer=init_ops.truncated_normal_initializer(mean=0.0, stddev=0.01),
                            activation_fn=None, # Leave as ReLu?
                            normalizer_fn=None,
                            scope='conv10') # VALID?
        net = layers.avg_pool2d(net, [13, 13], stride=1, scope='pool10')                
        net = array_ops.squeeze(net, [1, 2], name='logits')
        logits = utils.collect_named_outputs(end_point_collection,
                                             sc.name + '/logits',
                                             net)
        end_points = utils.convert_collection_to_dict(end_point_collection)

  return logits, end_points

# For CIFAR-10
def inference24(inputs,
                num_classes=10,
                is_training=True,
                scope='squeezenet',
                weight_decay=WEIGHT_DECAY,
                batch_norm_decay=BATCHNORM_MOVING_AVERAGE_DECAY):
  with vs.variable_scope('squeezenet', values=[inputs]) as sc:
    end_point_collection = sc.original_name_scope + '_end_points'
    with arg_scope(squeezenet_arg_scope(is_training, weight_decay, batch_norm_decay)):
      with arg_scope([fire_module, layers.conv2d,
                            layers.max_pool2d, layers.avg_pool2d],
                          outputs_collections=[end_point_collection]):
        net = layers.conv2d(inputs, 64, [3, 3], stride=1, padding='VALID', scope='conv01') # VALID?
        net = layers.max_pool2d(net, [3, 3], stride=1, scope='pool01')
        net = fire_module(net, 16, 64, scope='fire02')
        net = fire_module(net, 16, 64, scope='fire03')
        net = layers.max_pool2d(net, [3, 3], stride=1, scope='pool03')
        net = fire_module(net, 32, 128, scope='fire04')
        net = fire_module(net, 32, 128, scope='fire05')
        net = layers.max_pool2d(net, [3, 3], stride=1, scope='maxpool05')
        net = fire_module(net, 48, 192, scope='fire06')
        net = fire_module(net, 48, 192, scope='fire07')
        net = fire_module(net, 64, 256, scope='fire08')
        net = fire_module(net, 64, 256, scope='fire09')
        net = layers.conv2d(net, num_classes, [1, 1], 
                            stride=1, padding='VALID',
                            weights_initializer=init_ops.truncated_normal_initializer(mean=0.0, stddev=0.01),
                            activation_fn=None, # Leave as ReLu?
                            normalizer_fn=None,
                            scope='conv10') # VALID?
        net = layers.avg_pool2d(net, [16, 16], stride=1, scope='pool10')
        net = array_ops.squeeze(net, [1, 2], name='logits')
        logits = utils.collect_named_outputs(end_point_collection,
                                              sc.name + '/logits',
                                              net)
        end_points = utils.convert_collection_to_dict(end_point_collection)

  return logits, end_points

def squeezenet_arg_scope(is_training=True, weight_decay=WEIGHT_DECAY, bn_decay=BATCHNORM_MOVING_AVERAGE_DECAY):
  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': bn_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      'is_training': is_training,
      'center': True,
      'scale': True,
      #'updates_collections': None,
  }
  # Parameters for Conv2d weights
  weight_reg = regularizers.l2_regularizer(weight_decay)

  with arg_scope([layers.conv2d],
                  activation_fn=nn_ops.relu,
                  normalizer_fn=layers.batch_norm,
                  normalizer_params=batch_norm_params,
                  weights_initializer=init_ops.glorot_uniform_initializer(),
                  weights_regularizer=weight_reg,
                  biases_initializer=init_ops.zeros_initializer()) as sc:
      return sc