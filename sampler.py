from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

from glob import glob

import sys
import numpy as np
import tensorflow as tf

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

        
class LatentSampler(object):
    # Provide Gaussian feed for latent space z of Generative Network
    
    def __init__(self, shape, mean, variance):
        self.shape = shape
        self.mean = mean
        self.variance = variance
    
        self.batch_calls = 0
        self.samples_provided = 0
       
    
    def next_batch(self, N):
        
        self.batch_calls += 1
        self.samples_provided += N
        
        return (np.random.randn(N, *self.shape) + self.mean) / self.variance
    
            

class DataSampler():
    # Loads images and provides batch samples for network
    
    def __init__(self, image_size=32, pre_load=True):
        # For statistics and stuff
        self.batch_calls = 0
        self.samples_provided = 0
        self.next_sample = 0
        
        self.pre_load = pre_load
        self.image_size = image_size
        
        if image_size == 32:
            filenames = glob('data_source/32x32/*')
        elif image_size == 128:
            filenames = glob('data_source/128x128/*')
        elif image_size == 256:
            filenames = glob('data_source/256x256/*')
        else:
            raise ValueError('No dataset available for size %i' % size)
            
        
        self.filenames = filenames
        
        if pre_load:
            self.data = self.read_from_source(len(filenames), verbose=True)    
            
    def read_file_format(self, filename_queue):
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        image = tf.image.decode_png(value, channels=3)
        
        #PREPROCESS COMES HERE...
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image.set_shape([self.image_size, self.image_size, 3])
        return image
        
    def read_from_source(self, N, verbose=False):
        filename_queue = tf.train.string_input_producer(self.filenames, 
                                                        shuffle=True)
        images = self.read_file_format(filename_queue)
        
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            batch = []
            for n in xrange(N):
                image = images.eval()
                batch.append(image)
                if verbose: 
                    print('\rReading dataset from source... %i/%i' % (n+1, N), 
                          end='')
                    sys.stdout.flush()

            coord.request_stop()
            coord.join(threads)
        return np.array(batch)
        
    def vanilla_next_batch(self, N):
        if N > len(self.filenames):
            raise ValueError(
                'Required batch size %i is larger than whole dataset'%N)
        self.batch_calls += 1
        self.samples_provided += N
        self.next_sample = (self.samples_provided+1)%len(self.filenames)
        
        if self.pre_load:
            return self.data[self.next_sample:self.next_sample + N]
        else:
            return self.read_from_source(N)
    
    def get_next_sample_name(self):
        return self.filenames[self.next_sample]
    
    def tensorflow_next_batch(self, N, read_threads=4):
        
        # source of base:
        # https://www.tensorflow.org/versions/r0.12/how_tos/reading_data/index.html#Feeding
        
        filename_queue = tf.train.string_input_producer(
            self.filenames, shuffle=True)
        
        # min_after_dequeue defines how big a buffer we will randomly sample
        #   from -- bigger means better shuffling but slower start up and more
        #   memory used.
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size 
        
        min_queue_examples = 50 * N
        capacity = min_queue_examples + 3 * N
        image = self.read_file_format(filename_queue)
        
        # Multithread solution
        #image_list = [self.read_file_format(filename_queue)for _ in range(read_threads)]
        
        image_batch = tf.train.shuffle_batch(
            tensors=[image, ], 
            #shapes=[image_shape, ]*read_threads,
            batch_size=N, capacity=capacity, 
            min_after_dequeue=min_queue_examples)
        
        return image_batch
    

