'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

import time
from datetime import timedelta

import os
import psutil

import dataset
import config as conf

process = psutil.Process(os.getpid())

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

graph = tf.Graph()
saver = tf.train.Saver()

data = dataset.read_train_sets(conf.train_path, conf.img_size, conf.classes, validation_size=conf.validation_size)


# Parameters
learning_rate = 0.0001
training_iters = conf.iterations
batch_size = conf.batch_size
display_step = 1

# Network Parameters
n_input = conf.img_size # MNIST data input (img shape: 28*28)
# n_input = 224 # MNIST data input (img shape: 28*28)
n_steps = conf.img_size # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = conf.num_classes # MNIST total classes (0-9 digits)


print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
# x = tf.placeholder("float", [None, conf.img_size_flat])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

with graph.name_scope("final_layer") :
    rnnPred = RNN(x, weights, biases)
pred = rnnPred

y_pred = tf.nn.softmax(pred, name="final_result")

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
memory = []
cpu = []
with tf.Session() as sess:
    sess.run(init)
    # Keep training until reach max iterations
    # Start-time used for printing time-usage below.
    start_time = time.time()
    x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)

    cap = 0
    accurac = [0] * conf.num_channels
    while cap < conf.num_channels   :
        step = 1
        valor = 0
        while step * batch_size < training_iters:
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            print('-----------{0}-----------------'.format(cap))
            x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

            # x_batch = x_batch.reshape(batch_size, conf.img_size_flat)
            # x_valid_batch = x_valid_batch.reshape(batch_size, conf.img_size_flat)

            # x_batch = x_batch.reshape((batch_size, n_steps, n_input))

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: x_batch[:, :, :, int(cap)], y: y_true_batch})
            
            val_loss  = 0

            if step % display_step == 0:
                # Calculate batch Accuracycy
                acc = sess.run(accuracy, feed_dict={x: x_batch[:, :, :, int(cap)], y: y_true_batch})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: x_batch[:, :, :, int(cap)], y: y_true_batch})
                val_loss = loss

                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                memory.append(process.memory_info().rss)
                cpu.append((process.cpu_percent()))
            step += 1

        tf.add_to_collection('final_result', y_pred)

        # name = ('model_' + str(training_iters) + '_' + str(val_loss)) 
        # Save model weights to disk
        # save_path = saver.save(session, conf.model_path + name)
        # tf.train.export_meta_graph(conf.model_path + name)
        #write_predictions(test_images, test_ids)
        # print("Model saved in file: %s" % save_path)

        print("Optimization Finished!")
        end_time = time.time()
        end_time2 = time.time()
        # Calculate accuracy for 128 mnist test images
        test_len = 2
        test_images, test_ids = dataset.read_test_set(conf.test_path, conf.img_size)
        # test_images = test_images.reshape((-1, n_steps, n_input))
        # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        # test_label = mnist.test.labels[:test_len]
        accurac[cap] = sess.run(accuracy, feed_dict={x: test_images[:test_len][:, :, :, int(cap)], y: y_true_batch[:test_len]})
        print("Testing Accuracy:", accurac[cap])
        # Difference between start and end-times.
        time_dif = end_time - start_time
        time_dif2 = end_time2 - start_time
        # Print the time-usage.
        print("cap {0}".format(cap))
        print("Time elapsed1: " + str(timedelta(seconds=int(round(time_dif)))))
        print("Time elapsed2:" + str(timedelta(seconds=int(round(time_dif2)))))
        print("\n\n")

        cap += 1
        
        # memory = sum(memory) / len(memory)
        # cpu = sum(cpu) / len(cpu)
        # print(cpu,memory)
print("Average:")
avgAcc = (accurac[0]+accurac[1]+accurac[2])/3
print(avgAcc)
