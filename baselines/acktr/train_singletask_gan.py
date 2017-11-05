from collections import deque
import gym
import joblib
import numpy as np
import tensorflow as tf
import os, logging
import math
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.acktr.policies import CnnPolicy
from baselines.common import set_global_seeds
from baselines import logger
from baselines import bench
from baselines.acktr.utils import GeneratorNoiseInput, DataDistribution
from baselines.acktr.gan_models import CnnGenerator, CnnDiscriminator

class GAN():
    def __init__(self):
        # WGAN hyperparams
        batch_size = 100
        discriminator_steps = 5
        generator_steps = 1
        input_noise_size = 100
        generator_learning_rate = 0.0003
        discriminator_learning_rate = 0.001
        noise_size = 100
        generator_noise = GeneratorNoiseInput(noise_size)
        _lambda = 10
        adam_beta1 = 0.5
        adam_beta2 = 0.9
        num_epochs = 1 #epochs of generator updates before getting next simulation batch
        shuffle_data = False
        summary_interval = 10
        checkpoint_interval = 10
        
        # RL hyperparams
        total_timesteps=1e6 * args.million_frames
        num_simulation = 1000 #number of time steps to simulate expert before training GAN
        nprocs = 1
        nenvs = 1
        nstack = 4
        nsteps = 1
        # Parameters for testing generator in game - not yet implemented
        generator_simulation_interval = 20
        generator_simulation_steps = 100
        
        env = gym.make(args.env)
        if logger.get_dir():
            env = bench.Monitor(env, os.path.join(logger.get_dir(), "sample.monitor.json"))
        gym.logger.setLevel(logging.WARN)
        env = wrap_deepmind(env)

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
        expert_pi = tf.placeholder(tf.float32, [batch_size, action_shape])
        
        with tf.variable_scope("Generator"):
            generator = CnnGenerator(sess, obvs, action_space, noise_size, batch_size, nstack)
        with tf.variable_scope("Expert"):
            expert = CnnPolicy(sess, state_space, action_space, nenvs, nsteps, nstack)
        with tf.variable_scope("Discriminator"):
            discriminator_generator = CnnDiscriminator(sess, obvs, generator.pi, batch_size, nstack)
            discriminator_expert = CnnDiscriminator(sess, obvs, expert_pi, batch_size, nstack, reuse=True)
            
            alpha = tf.random_uniform([batch_size,1])
            differences = generator.pi - expert_pi
            interpolates = expert_pi + alpha*differences
            # print(interpolates.get_shape())
            discriminator_gradient = CnnDiscriminator(sess, obvs, interpolates, batch_size, nstack, reuse=True)
            
            results = discriminator_gradient.discriminator_decision
            # print(results.get_shape())
            
            gradients = tf.gradients(results, [interpolates])[0]
            # print(gradients.get_shape())
            slopes = tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(gradients,[batch_size,-1])), reduction_indices=[1]))
            gradient_penalty = _lambda*tf.reduce_mean((slopes-1.)**2)
        
        generator_loss = -tf.reduce_mean(discriminator_generator.discriminator_decision)
        wasserstien_distance = tf.reduce_mean(discriminator_generator.discriminator_decision)-tf.reduce_mean(discriminator_expert.discriminator_decision)
        discriminator_loss = wasserstien_distance + gradient_penalty
        
        vars = tf.trainable_variables()
        generator_params = [v for v in vars if v.name.startswith('Generator/')]
        discriminator_params = [v for v in vars if v.name.startswith('Discriminator/')]
        expert_params = [v for v in vars if v.name.startswith('Expert/')]
        
        generator_opt = tf.train.AdamOptimizer(generator_learning_rate, beta1=adam_beta1, beta2=adam_beta2)
        generator_grads = generator_opt.compute_gradients(generator_loss, var_list=generator_params)
        generator_opt_op = generator_opt.apply_gradients(generator_grads)
        
        discriminator_opt = tf.train.AdamOptimizer(discriminator_learning_rate, beta1=adam_beta1, beta2=adam_beta2)
        discriminator_grads = discriminator_opt.compute_gradients(discriminator_loss, var_list=discriminator_params)
        discriminator_opt_op = discriminator_opt.apply_gradients(discriminator_grads)
    
        discriminator_data_dis = DataDistribution(shuffle_data)
        generator_data_dis = DataDistribution(shuffle_data)
        
        tf.summary.scalar('Wasserstein Distance',wasserstien_distance)
        tf.summary.scalar('D_Loss',discriminator_loss)
        tf.summary.scalar('D_Gradient_Penalty',gradient_penalty)
        tf.summary.scalar('D_Expert_Score',tf.reduce_mean(discriminator_expert.discriminator_decision))
        tf.summary.scalar('D_Generator_Score',tf.reduce_mean(discriminator_generator.discriminator_decision))
        tf.summary.scalar('G_Loss',generator_loss)
        
        def train():
            def update_obs(state_u, obs_u):
                obs_u = np.reshape( obs_u, state_u.shape[0:3] )
                state_u = np.roll(state_u, shift=-1, axis=3)
                state_u[:, :, :, -1] = obs_u
                return state_u
            tf.global_variables_initializer().run(session=sess)
            saver = tf.train.Saver(max_to_keep=2)
            summary = tf.summary.merge_all()
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
                print("Running step {} of {}".format(i,total_steps))
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

                discriminator_data_dis.update_data(data)
                generator_data_dis.update_data(data)
                for j in range(math.ceil(num_epochs * num_simulation / (batch_size * generator_steps))):
                    avg_distance = []
                    for k in range(discriminator_steps):
                        distance, _ = sess.run([wasserstien_distance, discriminator_opt_op], feed_dict=self.build_feed_dict(discriminator_data_dis))
                        avg_distance.append(distance)
                    for k in range(generator_steps):
                        sess.run(generator_opt_op, feed_dict=self.build_feed_dict(generator_data_dis))
                print("Current Wasserstein distance: {}".format(np.average(avg_distance)))
                if i%checkpoint_interval == 0 or (i+1) == total_steps:
                    checkpoint_file = os.path.join(args.ckpt_dir, 'checkpoint')
                    saver.save(sess, checkpoint_file, global_step=i)
                if i%summary_interval == 0 or (i+1) == total_steps:
                    summary_str = sess.run(summary, feed_dict=self.build_feed_dict(generator_data_dis))
                    summary_writer.add_summary(summary_str, i)
                    summary_writer.flush()
            
        self.train = train
        self.generator_noise = generator_noise
        self.generator = generator
        self.obvs = obvs
        self.expert_pi = expert_pi
        self.batch_size = batch_size
    
    def build_feed_dict(self, data_func):
        obs_data, pi_data, epochs = data_func.sample(self.batch_size)
        generator_noise = self.generator_noise.sample(self.batch_size)
        feed_dict = {}
        feed_dict[self.obvs] = obs_data
        feed_dict[self.expert_pi] = pi_data
        feed_dict[self.generator.noise] = generator_noise
        
        return feed_dict
            
            
def run():
    model = GAN()
    model.train()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip', type=int, default=1)
    # parser.add_argument('--save_interval', help='How often to save model', type=int, default=1000)
    parser.add_argument('--ckpt_dir', help='GAN model save directory', default='.')
    parser.add_argument("--load_model_path", help="Loading the saved expert")

    global args
    args = parser.parse_args()    
    run()