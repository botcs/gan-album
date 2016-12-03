from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

from glob import glob

import sys
import numpy as np
import tensorflow as tf



def _weight_variable(shape, dev=0.1, name='W'):
    initial = tf.truncated_normal(stddev=dev, shape=shape)
    #return tf.get_variable(name=name, initializer=initial)
    return tf.Variable(initial,)

def _bias_variable(size, name='b'):
    initial = tf.constant(value=0.1, shape=[size])
    #return tf.get_variable(name=name, initializer=initial)
    return tf.Variable(initial)

def linear(input, out_dim, scope_name='linear', ret_var=False):
    with tf.variable_scope(scope_name) as scope:
        in_dim = input.get_shape().as_list()[-1]
        mat_shape = in_dim, out_dim
        W = _weight_variable(mat_shape)

        b = _bias_variable(out_dim)
        out = tf.matmul(input, W) + b
    
    if ret_var: return out, W, b
    
    return out

def conv(input, kernel_size, depth, pad, scope_name='convolution', ret_var=False):
    with tf.variable_scope(scope_name) as scope:
        padding = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
        padded_input = tf.pad(input, padding, "CONSTANT")

        in_D = padded_input.get_shape().as_list()[-1]

        K = kernel_size
        D = depth

        kernel = _weight_variable([K, K, in_D, D], 'K')
        bias = _bias_variable(D)

        lin = tf.nn.conv2d(
            padded_input, kernel, strides=[1, 1, 1, 1], padding='SAME')
        out = lin+bias

    if ret_var: return out, K, bias
    return out
    
def conv_T(input, kernel_size, depth, 
           stride=2, scope_name='transposed_convolution', 
           ret_var=False):
    with tf.variable_scope(scope_name) as scope:
        batch, W_in, H_in, D_in = input.get_shape().as_list()
        W_out = W_in * stride
        H_out = H_in * stride
        K = kernel_size
        D_out = depth
        
        # conv2d_transposed requires this workaround for static batch size
        batch = tf.shape(input)[0]
        output_shape = [batch, H_out, W_out, D_out]
        kernel = _weight_variable([K, K, D_out, D_in], dev=0.01, name='K')
        bias = _bias_variable(D_out)
        lin = tf.nn.conv2d_transpose(
            value=input, 
            filter=kernel,
            output_shape=output_shape,
            strides=[1, stride, stride, 1], 
            padding='SAME')
        
        # conv2d_transposed discards the shape also... what a shame!
        out = tf.reshape(lin + bias, output_shape)
    if ret_var: return out, kernel, bias    
    return out
