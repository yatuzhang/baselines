import numpy as np
import tensorflow as tf
from baselines.acktr.utils import conv, fc, dense, conv_to_fc, sample, kl_div

class CnnGenerator(object):
    def __init__(self, sess, obvs, ac_space, noise_size, batch_size, nstack, reuse=False):
        nact = ac_space.n
        noise = tf.placeholder(tf.float32, [batch_size, noise_size])
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(obvs, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=32, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            generator_input = tf.concat([h3, noise],1)
            h4 = fc(generator_input, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x:x)
            vf = fc(h4, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful
        
        def step(feed_dict, *_args, **_kwargs):
            a, v = sess.run([a0, v0], feed_dict)
            return a, v, [] #dummy state

        self.pi = pi
        self.vf = vf
        self.noise = noise
        self.step = step

class CnnDiscriminator(object):
    def __init__(self, sess, obvs, action_pi, batch_size, nstack, reuse=False):
        with tf.variable_scope("model", reuse=reuse):
            # Build a feature detection layer with the convolutional part
            h = conv(tf.cast(obvs, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=32, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            # Concatenate the features with the input action
            discriminator_input = tf.concat([h3, action_pi],1)
            h4 = fc(discriminator_input, 'fc1', nh=512, init_scale=np.sqrt(2))
            discriminator_decision = fc(h4, 'decision', 1, act=lambda x:x)
            
        self.initial_state = [] #not stateful - ???

        self.discriminator_decision = discriminator_decision

# Could also have a CNN feature detector that feeds into both a generator and a discriminator