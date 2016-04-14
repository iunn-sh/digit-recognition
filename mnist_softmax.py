# Copyright 2015 Google Inc. All Rights Reserved.
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

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
import input_data
import recognizer

import numpy as np
import pandas as pd
import tensorflow as tf
from os import path

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')


num_examples = 10000
epochs_completed = 0
index_in_epoch = 0

def next_batch(images,labels,batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    global num_examples 
    global epochs_completed 
    global index_in_epoch 

    if fake_data:
      fake_image = [1] * 784
      
      fake_label = [1] + [0] * 9
      
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]

    start = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch > num_examples:
      # Finished epoch
      epochs_completed += 1
      # Shuffle the data
      perm = np.arange(num_examples)
      np.random.shuffle(perm)
      images = images[perm]
      labels = labels[perm]
      # Start next epoch
      start = 0
      index_in_epoch = batch_size
      assert batch_size <= num_examples
    end = index_in_epoch
    #print ('%s - %s' %(start,end))
    return images[start:end], labels[start:end]

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def weight_variable(shape):
 	initial = tf.truncated_normal(shape, stddev=0.1)
 	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
  	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

ap_dir = path.dirname(__file__)  # <-- absolute dir the script is in

#prepare images
train_images = pd.read_csv(path.join(ap_dir, 'data/train.csv'),header=None)
print('data({0[0]},{0[1]})'.format(train_images.shape))
print (train_images.head())
train_images = train_images.iloc[:,1:].values
# convert from [0:255] => [0.0:1.0]
train_images = np.multiply(train_images, 1.0 / 255.0)
print('images({0[0]},{0[1]})'.format(train_images.shape))

# prepare label
train_labels = pd.read_csv(path.join(ap_dir, 'data/label.csv'),header=None)
train_labels = train_labels[[1]].values.ravel()
print('train_labels({0})'.format(len(train_labels)))
print ('train_labels[{0}] => {1}'.format(1,train_labels[1]))

labels_count = np.unique(train_labels).shape[0]
train_labels = dense_to_one_hot(train_labels, labels_count)
train_labels = train_labels.astype(np.uint8)

print('labels({0[0]},{0[1]})'.format(train_labels.shape))
print ('labels[{0}] => {1}'.format(1,train_labels[1]))

validation_images = train_images[:2000]
validation_labels = train_labels[:2000]

print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))
print('validation_images({0[0]},{0[1]})'.format(validation_labels.shape))

test_images = [[]]
test_labels = [[]]


sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Train
tf.initialize_all_variables().run()
for i in range(1000):
  batch_xs, batch_ys = next_batch(train_images,train_labels,100)
  train_step.run({x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
print(accuracy.eval({x: validation_images, y_: validation_labels}))



