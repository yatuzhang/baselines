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
    # check to ensure full action space is used
    assert env.action_space.n == 18, "amount of actions in action space is :{}, not equal to full action space".format(env.action_space.n)
    if logger.get_dir():
        env = bench.Monitor(env, os.path.join(logger.get_dir(), "sample.monitor.json"))
    gym.logger.setLevel(logging.WARN)
    env = wrap_deepmind(env)

    tf.reset_default_graph()
    set_global_seeds(0)

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

    policy=CnnPolicy
    model = Model(policy, state_space, action_space, nenvs, total_timesteps, nprocs=nprocs, nstack=nstack)
    model.load(args.load_model_path)
    act = model.step_model
    
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
        states = model.initial_state

        obs = env.reset()
        done = False
        episode_reward = 0
        debug = 0
        # Add a step counter ensuring a max amount of game run time
        game_steps_played = 0
        while not done:
            if debug % 100 == 0:
                print("In loop {}".format(debug))
            debug = debug + 1
            if args.save_ani:
                im = plt.imshow(env.render(mode='rgb_array'))
                ims.append([im])
            state = update_obs(state,obs)
            actions, values, features, states = act.step_w_features(state, states, [done])
            obs, rew, done, _ = env.step(actions[0])
            
            episode_reward += rew
            if args.save_trajectory:
                states_arr.append(state)
                features_arr.append(features)
                action_values.append(actions)
            game_steps_played += 1
            if(game_steps_played == 5000):
                print("Have run {} game steps without lossing the game. Exiting the game episode...".format(5000))
                break
        if args.save_ani:
            ani = animation.ArtistAnimation(fig, ims, interval=20)
            ani.save(os.path.join(args.save_ani_path,'episode_{}.mp4'.format(episode)))
            ims=[]
        rewards.append(episode_reward)
        logger.record_tabular("Episode", episode)
        logger.record_tabular("Immediate Reward", float(episode_reward))
        logger.record_tabular("Running Average", float(np.mean(rewards)))
        logger.dump_tabular()
        print("<<<<<<<<<<<Finished Generating One Episode of Expert Model Game Playing!>>>>>>>>>>>>")
        episode += 1
        
    if args.save_trajectory:
        np.savez("data", states=states_arr,features=features_arr, action_values=action_values)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--max_episodes", default="1500", type=int, help="Maximum number of episodes to play.")
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument("--load_model_path", help="Loading the saved model")
    parser.add_argument("--save_trajectory", type=bool, default=False, help="Saves an animation of the run")
    parser.add_argument("--save_ani", type=bool, default=False, help="Saves an animation of the run")
    parser.add_argument("--save_ani_path", default='./', help="directory to save animation")

    global args
    args = parser.parse_args()    
    run()
