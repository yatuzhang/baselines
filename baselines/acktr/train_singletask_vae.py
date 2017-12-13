from collections import deque
import gym
import joblib
import numpy as np
import tensorflow as tf
import os, logging
import math
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common import set_global_seeds
from baselines.acktr.policies import CnnPolicy
from baselines import logger
from baselines import bench
from baselines.acktr.utils import DataDistribution, plot_vae
from baselines.acktr.vae_models import VAEModel

class VAE():
    def __init__(self):
        # WGAN hyperparams
        batch_size = 100
        learning_rate = 0.001
        num_epochs = 1 #epochs of updates before getting next simulation batch
        shuffle_data = True
        summary_interval = 10
        checkpoint_interval = 10
        image_interval = 10
        num_Gaussians = 10
        
        # RL hyperparams
        total_timesteps = 1e6 * args.million_frames
        total_timesteps = int(total_timesteps / 4 * 1.1) 
        num_simulation = 1000 #number of time steps to simulate expert before training GAN
        nprocs = 1
        nenvs = 1
        nstack = 4
        nsteps = 1
        
        env = gym.make(args.env,)
        if logger.get_dir():
            env = bench.Monitor(env, os.path.join(logger.get_dir(), "sample.monitor.json"))
        gym.logger.setLevel(logging.WARN)
        env = wrap_deepmind(env, clip_rewards=False)

        tf.reset_default_graph()
        set_global_seeds(0)

        state_space = env.observation_space
        action_space = env.action_space
        action_shape = env.action_space.n
        nh, nw, nc = state_space.shape
        batch_state_shape = (nenvs*nsteps, nh, nw, nc*nstack)
        batch_state_reshape = (nh, nw, nc*nstack)
        batch_obs_shape = (batch_size, nh, nw, nc*nstack)
        
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        
        # Create placeholders
        obvs = tf.placeholder(tf.uint8, batch_obs_shape)
        
        with tf.variable_scope("Generator"):
            vae = VAEModel(sess, obvs, batch_size, nstack, num_Gaussians)
        with tf.variable_scope("Expert"):
            expert = CnnPolicy(sess, state_space, action_space, nenvs, nsteps, nstack)
        
        obvs_flat = tf.reshape(tf.cast(obvs, tf.float32)/255., [batch_size,-1])
        reconstruction_flat = tf.reshape(vae.reconstruction, [batch_size,-1])
        
        reconstruction_loss = -tf.reduce_sum(obvs_flat * tf.log(1e-10 + reconstruction_flat) + (1-obvs_flat) * tf.log(1e-10 + 1 - reconstruction_flat), 1)
        latent_loss = -0.5 * tf.reduce_sum(1 + vae.z_log_sigma_sq - tf.square(vae.z_mean) - tf.exp(vae.z_log_sigma_sq), 1)
        loss = tf.reduce_mean(reconstruction_loss + latent_loss)
                
        vars = tf.trainable_variables()
        vae_params = [v for v in vars if v.name.startswith('Generator/')]
        expert_params = [v for v in vars if v.name.startswith('Expert/')]
        
        opt = tf.train.AdamOptimizer(learning_rate)
        grads = opt.compute_gradients(loss, var_list=vae_params)
        opt_op = opt.apply_gradients(grads)
    
        data_dis = DataDistribution(shuffle_data)
        
        #Defining training statistics
        tf.summary.scalar('NLL',loss, collections=['train'])

        def train():
            def update_obs(state_u, obs_u):
                obs_u = np.reshape( obs_u, state_u.shape[0:3] )
                state_u = np.roll(state_u, shift=-1, axis=3)
                state_u[:, :, :, -1] = obs_u
                return state_u
            tf.global_variables_initializer().run(session=sess)
            saver = tf.train.Saver(max_to_keep=2)
            summary_train = tf.summary.merge_all('train')
            summary_evaluate = tf.summary.merge_all('evaluate')

            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
            summary_writer = tf.summary.FileWriter(args.ckpt_dir, sess.graph)

            loaded_params = joblib.load(args.load_model_path)
            restores = []
            for p, loaded_p in zip(expert_params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            
            act = expert.step_w_pi
            state = np.zeros(batch_state_shape, dtype=np.uint8)
            states = expert.initial_state

            obs = env.reset()
            done = False
            
            total_steps = math.ceil(total_timesteps / num_simulation)
            
            for i in range(total_steps):
                logger.log("Running step {} of {}".format(i,total_steps))

                # Get expert data
                data = {}
                data['obs'] = []
                data['pi'] = []
                
                for l in range(num_simulation):
                    if done:
                        state = np.zeros(batch_state_shape, dtype=np.uint8)
                        states = expert.initial_state

                        obs = env.reset()
                        done = False
                        
                    state = update_obs(state,obs)
                    actions, values, pi, states = act(state, states, [done])
                    data['obs'].append(state.reshape(batch_state_reshape))
                    data['pi'].append(pi.reshape(action_shape))
                    obs, rew, done, _ = env.step(actions[0])

                data_dis.update_data(data)
                
                for j in range(math.ceil(num_epochs * num_simulation / batch_size)):
                    _, latest_loss = sess.run([opt_op, loss], feed_dict=self.build_feed_dict(data_dis))
                logger.log("Current Loss: {}".format(latest_loss))
                
                if i%checkpoint_interval == 0 or (i+1) == total_steps:
                    checkpoint_file = os.path.join(args.ckpt_dir, 'checkpoint')
                    saver.save(sess, checkpoint_file, global_step=i)
                    
                if i%summary_interval == 0 or (i+1) == total_steps:
                    summary_str = sess.run(summary_train, feed_dict=self.build_feed_dict(data_dis))
                    summary_writer.add_summary(summary_str, i)
                    summary_writer.flush()
                    
                if i%summary_interval == 0 or (i+1) == total_steps:
                    feed_dict = self.build_feed_dict(data_dis)
                    reconstruction = sess.run(vae.reconstruction, feed_dict=feed_dict)
                    for k in range(3):
                        real_image = np.split(feed_dict[self.obvs][k],nc*nstack, axis=-1)
                        real_image = np.concatenate(real_image,axis=1)
                        real_image = real_image.reshape(nh,nw*nstack*nc)
                        real_image = real_image/255.
                        generated_image = np.split(reconstruction[k],nc*nstack, axis=-1)
                        generated_image = np.concatenate(generated_image,axis=1)
                        generated_image = generated_image.reshape(nh,nw*nstack*nc)
                        plot_vae(real_image, generated_image, i, k, vmin=0, vmax=1)
            
        self.train = train
        self.obvs = obvs
        self.batch_size = batch_size
    
    def build_feed_dict(self, data_func):
        obs_data, pi_data, epochs = data_func.sample(self.batch_size)
        feed_dict = {}
        feed_dict[self.obvs] = obs_data
        
        return feed_dict
            
            
def run():
    model = VAE()
    model.train()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip', type=int, default=10)
    # parser.add_argument('--save_interval', help='How often to save model', type=int, default=1000)
    parser.add_argument('--ckpt_dir', help='GAN model save directory', default='.')
    parser.add_argument("--load_model_path", help="Loading the saved expert")

    global args
    args = parser.parse_args()    
    run()