

#TEST CODE FOR SERVER
import time
start_time = time.time()
print('IMPORTING', end="\t", flush=True)
import sys
import argparse
import numpy as np 
import tensorflow as tf
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)



import tensorflow as tf
import numpy as np
from ops import *
import sampler
print(time.time() - start_time)
start_time=time.time()

print('NETWORK INITIALIZING', end="\t", flush=True)
def generator(out_dim=[32, 32], reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    
    
    z = tf.placeholder(dtype=tf.float32, shape=[None, 100], name='z')
    D = max(out_dim) * 16
    
    h0 = linear(z, out_dim=out_dim[0]//16 * out_dim[1]//16 * D, name='projection')
    h0 = tf.reshape(h0, [-1, out_dim[0]//16, out_dim[1]//16, D])
    
    h1 = conv_T(h0, kernel_size=5, depth=D//2, name='conv1')
    h1 = tf.nn.relu(h1)
    
    h2 = conv_T(h1, kernel_size=5, depth=D//4, name='conv2')
    h2 = tf.nn.relu(h2)
    
    h3 = conv_T(h2, kernel_size=5, depth=D//8, name='conv3')
    h3 = tf.nn.relu(h3)
    
    G = conv_T(h3, kernel_size=5, depth=3, name='conv4')
    G = tf.nn.relu(G)
    
    return G
    
    
def discriminator(image, start_depth, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    
    
    D = start_depth
    
    h0 = conv(image, kernel_size=5, depth=D, name='conv1')
    h0 = lrelu(h0)
    
    h1 = conv(h0, kernel_size=5, depth=D*2, name='conv2')
    h1 = lrelu(h1)
    
    h2 = conv(h1, kernel_size=5, depth=D*4, name='conv3')
    h2 = lrelu(h2)
    
    h3 = conv(h2, kernel_size=5, depth=D*8, name='conv4')
    h3 = lrelu(h3)
    
    flat_dim = np.prod(h3.get_shape().as_list()[1:])
    h3_flat = tf.reshape(h3, [-1, flat_dim])
    D = linear(h3_flat, out_dim=1)
    
    return D    
    
    
with tf.variable_scope('G') as scope:
    G = generator()

with tf.variable_scope('D') as scope:
    im_dim = [32, 32, 3]
    im = tf.placeholder(dtype=tf.float32, shape=[None] + im_dim, name='im')
    D_real = discriminator(im, 512)
    D_fake = discriminator(G, 512, reuse=True)
            

with tf.Session() as sess:
    TRAIN_ITER = 1000

    D_PER_ROUND = 5
    G_PER_ROUND = 7

    D_BATCH = 8
    G_BATCH = 8

    z_feed = sampler.LatentSampler(mean=.5, variance=.1, shape=[100])
    y_feed = sampler.DataSampler(image_size=32, pre_load=False).tensorflow_next_batch(D_BATCH)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Modified code from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py#L100-L102
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_real, tf.ones_like(D_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake, tf.zeros_like(D_fake)))
    
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake, tf.ones_like(D_fake)))
    

    
    g_vars = [v for v in tf.trainable_variables() if v.name.startswith('G')]
    d_vars = [v for v in tf.trainable_variables() if v.name.startswith('D')]

    D_reg = 0
    for g_v in g_vars:
        D_reg += tf.nn.l2_loss(g_v)
    
    G_reg = 0
    for d_v in d_vars:
        G_reg += tf.nn.l2_loss(d_v)
    
    #print(D_reg)
    #print(G_reg)
    REG_FACTOR = 5e-4
        
    D_regloss = D_loss + REG_FACTOR * D_reg
    G_regloss = G_loss + REG_FACTOR * G_reg
    
    d_optim = tf.train.MomentumOptimizer(0.0002, 0.5) \
        .minimize(D_regloss, var_list=d_vars)
    g_optim = tf.train.MomentumOptimizer(0.0002, 0.5) \
        .minimize(G_regloss, var_list=g_vars)
    
    tf.global_variables_initializer().run()    
    
    saver = tf.train.Saver()
    print(time.time() - start_time)
    start_time = time.time()
    
   

    print('TRAINING...{} x D({}) <-> {} x G({})'.\
        format(D_PER_ROUND, D_BATCH, G_PER_ROUND, G_BATCH))
    for it in range(100):
        print('\n\nROUND:', it+1)
        start_time = time.time()
        for d_it in range(D_PER_ROUND):
            l, _ = sess.run([D_regloss, d_optim],
                            {'G/z:0':z_feed.next_batch(D_BATCH), 
                             'D/im:0':y_feed.eval()})
            print('DSICRIMINATOR\t%2d \ttime: \t%4.2f \t+++LOSS: %4.4f'\
                % (d_it+1, time.time() - start_time, l))
            start_time = time.time()
                                 
        for g_it in range(G_PER_ROUND):
            l, _ = sess.run([G_regloss, g_optim],
                            {'G/z:0':z_feed.next_batch(G_BATCH)})
            print('GENERATOR\t%2d \ttime: \t%4.2f \toooLOSS: %4.4f'\
                % (g_it+1, time.time() - start_time, l))
            start_time = time.time()
                        
        
        if (it+1) % 200 == 0: 
            save_path = saver.save(sess, "./large_checkpoint_{}.chkpt".format(it+1))
            print("Model saved in file: %s" % save_path)
    

    coord.request_stop()
    coord.join(threads)
    sess.close()    
                 
    print('DONE')           
