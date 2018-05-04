#!/usr/bin/python
#-*- coding:utf-8 -*-
import tensorflow as tf

Actions = 4
REPLACE_ITER = 4

class MultilayerNetwork(object):

    def __init__(self,name,input_width, input_height, nimages):

        self.name = name
        self.input_width = input_width
        self.input_height = input_height
        self.nimages = nimages

        self.input_image = tf.placeholder("float", [None, self.input_width, self.input_height, self.nimages])
        self.input_a = tf.placeholder(tf.int32, [None])
        self.input_y = tf.placeholder("float", [None])

        self.q, self.q_action, self.q_real, self.cost = self.build_network(self.name, "critic")
        self.q_, _ , self.q_real_, _ = self.build_network(self.name, "target")


    def weight_variable(self, shape, stddev=0.01):

        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial)

    def bias_variable(self, shape, value=0.01):
        initial = tf.constant(value, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides = [1,2,2,1], padding = "SAME")

    def build_weight_biases(self, weights_shape):

        return self.weight_variable(weights_shape), self.bias_variable(weights_shape[-1:])

    def convolve_relu_pool(self, nn_input, weights_shape,stride = 4):

        W_conv,b_conv = self.build_weight_biases(weights_shape)
        h_conv = tf.nn.relu(self.conv2d(nn_input, W_conv, stride) + b_conv)
        return self.max_pool_2x2(h_conv)


    def build_network(self, name, scope):
        with tf.variable_scope(name):

            with tf.variable_scope(scope):

                h_pool1 = self.convolve_relu_pool(self.input_image,[8,8, self.nimages,32])
                h_pool2 = self.convolve_relu_pool(h_pool1, [4,4,32,64],2)
                h_pool3 = self.convolve_relu_pool(h_pool2, [3,3,64,64],1)

                h_pool3_flat = tf.reshape(h_pool3,[-1,256])

                W_fc1, b_fc1 = self.build_weight_biases([256,256])

                h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

                W_fc2, b_fc2 = self.build_weight_biases([256, Actions])
                readout = tf.matmul(h_fc1, W_fc2) + b_fc2

                readout_a = tf.argmax(readout, dimension =1)

                action_one_hot = tf.one_hot(self.input_a, Actions, 1.0, 0.0)

                readout_q = tf.reduce_sum(readout * action_one_hot, reduction_indices=1)
                cost = tf.reduce_mean(tf.square(self.input_y - readout_q))
                self.train_step = tf.train.RMSPropOptimizer(0.00025,0.95,0.95,0.01).minimize(cost)

                # readout  network输出的q值
                # readout_a 根据readout 选择的action
                # readout_q target_network输出的q值
                # cost 实际reward和真实值的差

        return readout, readout_a, readout_q, cost

'''
    def learn(self, state_batch,value_batch, action_batch):

        if self.replace_counter % REPLACE_ITER == 0:
            self.sess.run([tf.assign(t,e) for t,e in zip(self.ne_params, self.nt_params)])

        self.replace_counter += 1
        self.sess.run(self.train_step,feed_dict={self.input_image: state_batch, self.y: value_batch, self.a: action_batch})
'''


























