import numpy as np
import tensorflow as tf
from baselines.acktr.utils import conv, deconv, fc, dense, conv_to_fc, sample, kl_div

class VAEModel(object):
    def __init__(self, sess, obvs, batch_size, nstack, num_Gaussians, reuse=False):
        with tf.variable_scope("model", reuse=reuse):
            c_h = conv(tf.cast(obvs, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            c_h2 = conv(c_h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            c_h3 = conv(c_h2, 'c3', nf=32, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(c_h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            z_mean = fc(h4, 'z_mean', nh=num_Gaussians, act=lambda x:x)
            z_log_sigma_sq = fc(h4, 'z_sigma', nh=num_Gaussians, act=lambda x:x)
            
            eps = tf.random_normal((batch_size, num_Gaussians), 0, 1, dtype=tf.float32)
            z_sample = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
            
            g_h = fc(z_sample, 'g_fc1', nh=512, init_scale=np.sqrt(2))
            g_h2 = fc(g_h, 'g_fc2', nh=h3.get_shape()[-1], init_scale=np.sqrt(2))
            
            batch_size = tf.shape(g_h2)[0]
            g_c_h2 = tf.reshape(g_h2, [batch_size, c_h3.get_shape()[1].value, c_h3.get_shape()[2].value, c_h3.get_shape()[3].value])
            
            deconv_shape = [batch_size, c_h2.get_shape()[1].value, c_h2.get_shape()[2].value, c_h2.get_shape()[3].value]
            g_c_h3 = deconv(g_c_h2, 'g_c1', nf=64, rf=3, stride=1, output_size=deconv_shape, init_scale=np.sqrt(2))
            
            deconv_shape = [batch_size, c_h.get_shape()[1].value, c_h.get_shape()[2].value, c_h.get_shape()[3].value]
            g_c_h4 = deconv(g_c_h3, 'g_c2', nf=32, rf=4, stride=2, output_size=deconv_shape, init_scale=np.sqrt(2))
            
            deconv_shape = [batch_size, obvs.get_shape()[1].value, obvs.get_shape()[2].value, obvs.get_shape()[3].value]
            reconstruction_mean = deconv(g_c_h4, 'g_c3', nf=4, rf=8, stride=4, output_size=deconv_shape, init_scale=np.sqrt(2), act=tf.nn.sigmoid)

        self.z_mean = z_mean
        self.z_log_sigma_sq = z_log_sigma_sq
        self.z_sample = z_sample
        self.reconstruction = reconstruction_mean
