from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

from glob import glob

import sys
import numpy as np
import tensorflow as tf




def _weight_variable(shape, name='W'):
    initial = tf.truncated_normal(stddev=0.1, shape=shape)
    #return tf.get_variable(name=name, initializer=initial)
    return tf.Variable(initial)

def _bias_variable(size, name='b'):
    initial = tf.constant(value=0.1, shape=[size])
    #return tf.get_variable(name=name, initializer=initial)
    return tf.Variable(initial)

def _linear(input, out_dim):

    in_dim = input.get_shape().as_list()[-1]
    mat_shape = in_dim, out_dim
    W = _weight_variable(mat_shape)

    b = _bias_variable(out_dim)

    return tf.matmul(input, W) + b

def fc(input, width, scope_name):
    with tf.variable_scope(scope_name):
        lin = _linear(input, width)
    return tf.nn.relu(lin)

def conv(input, K, D, scope_name):
    
    in_D = input.get_shape().as_list()[-1]
    
    with tf.variable_scope(scope_name):
        kernel = _weight_variable([K, K, in_D, D], 'K')
        bias = _bias_variable(D)
        lin = tf.nn.conv2d(input, kernel, 
                           strides=[1, 1, 1, 1], 
                           padding='SAME')
        
        return tf.nn.relu(lin+bias)
    
def deconv(input, kernel_size, depth, scope_name, stride=2):
    
    batch, W_in, H_in, D_in = input.get_shape().as_list()
    
    W = stride * (W_in - 1) + kernel_size
    H = stride * (H_in - 1) + kernel_size
    K = kernel_size
    D = depth
    
    batch = tf.shape(input)[0]
    output_shape = [batch, H, W, D]
    with tf.variable_scope(scope_name):
        kernel = _weight_variable([K, K, D, D_in], 'K')
        bias = _bias_variable(D)
        lin = tf.nn.conv2d_transpose(
            value=input, 
            filter=kernel,
            output_shape=output_shape,
            strides=[1, 2, 2, 1], 
            padding='VALID')
    
    return tf.reshape(tf.nn.relu(lin + bias), output_shape)
    
