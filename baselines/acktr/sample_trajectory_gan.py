""" Use a pre-trained acktr model to play Breakout.
    To train: python3 ./run_atari.py
        You'll need to add "logger.configure(<some dir>)" to run_atari.py so it will save checkpoint files
    Then run this script with a checkpoint file as the argument
    A running average of the past 100 rewards will be printed
"""
from collections import deque
import gym
import cloudpickle
import numpy as np
import tensorflow as tf
import os, logging
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.acktr.acktr_disc import Model
from baselines.acktr.policies import CnnPolicy
from baselines.common import set_global_seeds, explained_variance
from baselines import logger
from baselines import bench
from baselines.acktr.utils import GeneratorNoiseInput
from baselines.acktr.gan_models import CnnGenerator

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update_obs(state, obs):
    obs = np.reshape( obs, state.shape[0:3] )
    state = np.roll(state, shift=-1, axis=3)
    state[:, :, :, -1] = obs
    return state

def run():
    env = gym.make(args.env)
    if logger.get_dir():
        env = bench.Monitor(env, os.path.join(logger.get_dir(), "sample.monitor.json"))
    gym.logger.setLevel(logging.WARN)
    env = wrap_deepmind(env)

    tf.reset_default_graph()
    set_global_seeds(0)
    
    noise_size = 100
    generator_noise = GeneratorNoiseInput(noise_size)

    total_timesteps=int(40e6)
    nprocs = 2
    nenvs = 1
    nstack = 4
    nsteps = 1
    nenvs = 1

    state_space = env.observation_space
    action_space = env.action_space
    print(env.unwrapped.get_action_meanings())
    nh, nw, nc = state_space.shape
    batch_state_shape = (nenvs*nsteps, nh, nw, nc*nstack)
    
    config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    obvs = tf.placeholder(tf.uint8, batch_state_shape)
    
    with tf.variable_scope("Generator"):
        generator=CnnGenerator(sess, obvs, action_space, noise_size, 1, nstack)
        
    ckpt = tf.train.get_checkpoint_state(args.load_model_dir)
    if ckpt and ckpt.model_checkpoint_path:
      vars_to_restore = tf.trainable_variables()
      restorer = tf.train.Saver(var_list=vars_to_restore)
      restorer.restore(sess, ckpt.model_checkpoint_path)
      restore = tf.train.latest_checkpoint(args.load_model_dir)
      print("Restored :{}".format(restore))      
    else:
      print("Error in restoring model")
      exit()
    
    episode = 1
    rewards= deque(maxlen=100)
    
    if args.save_ani:
        if args.max_episodes == 0 or args.max_episodes > 10:
            print("Are you sure you want to save that many episodes?")
            print("Exiting")
            exit()
        fig = plt.figure()
        plt.axis('off')
        ims = []
        
    if args.save_trajectory:
        states_arr = []
        features_arr = []
        action_values =[]

    while args.max_episodes == 0 or episode <= args.max_episodes:
        state = np.zeros(batch_state_shape, dtype=np.uint8)

        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if args.save_ani:
                im = plt.imshow(env.render(mode='rgb_array'))
                ims.append([im])
            state = update_obs(state,obs)
            
            # Build feed_dict
            feed_dict = {}
            feed_dict[obvs] = state
            feed_dict[generator.noise] = generator_noise.sample(1)
            
            actions, values, _ = generator.step(feed_dict)
            obs, rew, done, _ = env.step(actions[0])
            episode_reward += rew
            if args.save_trajectory:
                states_arr.append(state)
                features_arr.append(features)
                action_values.append(actions)
        if args.save_ani:
            ani = animation.ArtistAnimation(fig, ims, interval=20)
            ani.save(os.path.join(args.save_ani_path,'episode_{}.mp4'.format(episode)))
            ims=[]
        rewards.append(episode_reward)
        logger.record_tabular("Episode", episode)
        logger.record_tabular("Immediate Reward", float(episode_reward))
        logger.record_tabular("Running Average", float(np.mean(rewards)))
        logger.dump_tabular()
        episode += 1
        
    if args.save_trajectory:
        np.savez("data", states=states_arr,features=features_arr, action_values=action_values)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--max_episodes", default="1500", type=int, help="Maximum number of episodes to play.")
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument("--load_model_dir", help="Loading the saved model")
    parser.add_argument("--save_trajectory", type=bool, default=False, help="Saves an animation of the run")
    parser.add_argument("--save_ani", type=bool, default=False, help="Saves an animation of the run")
    parser.add_argument("--save_ani_path", default='./', help="directory to save animation")

    global args
    args = parser.parse_args()    
    run()