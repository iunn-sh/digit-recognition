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
from scipy import sparse
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

def appendHeader(header,data):
	out = []
	for i in range(header):
		row = []
		row.extend(header[i])
		row.extend(data[i])
		out.append(row)
	return out
		
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
train_header = train_labels.iloc[:,0:1].values
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

test_images = pd.read_csv(path.join(ap_dir, 'data/test.csv'),header=None)
print('test_images({0[0]},{0[1]})'.format(test_images.shape))
print (test_images.head())
test_head = test_images.iloc[:,0:1].values
test_images = test_images.iloc[:,1:].values

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)
print('test_images({0[0]},{0[1]})'.format(test_images.shape))
test_labels = [[]]


sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer


# multi layer model
#layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_ = tf.placeholder(tf.float32, [None, 10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train
tf.initialize_all_variables().run()

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(6000):
  batch = next_batch(train_images,train_labels,50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={ x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
print("validation_images accuracy %g"%accuracy.eval(feed_dict={
    x: validation_images, y_: validation_labels, keep_prob: 1.0}))

"""    

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
  batch_xs, batch_ys = next_batch(train_images,train_labels,100)
  train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
"""
#get prediction probability

pred_prob = y_conv.eval(feed_dict={x: test_images, keep_prob: 1.0}, session=sess)
pred_prob = pred_prob.astype(str)

print('pred_prob({0[0]},{0[1]})'.format(pred_prob.shape))
print(pred_prob)
print('test_head({0[0]},{0[1]})'.format(test_head.shape))
#print(test_head[1])
#print(train_header[1])
print(test_head)
out = np.hstack((test_head, pred_prob))
print(out[1])
np.savetxt('data/submit-tf-nn.csv', out, delimiter=',', header = '', fmt='%s')

#out = np.concatenate((np.array(test_head).T, np.array(pred_prob)), axis=1)
#print(out)
#get prediction
#prediction = tf.argmax(y,1)
#print (prediction.eval(feed_dict={x: validation_images}))

# Test trained model
