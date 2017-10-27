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

    while args.max_episodes == 0 or episode <= args.max_episodes:
        state = np.zeros(batch_state_shape, dtype=np.uint8)
        states = model.initial_state

        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state = update_obs(state,obs)
            actions, values, states = act.step(state, states, [done])
            obs, rew, done, _ = env.step(actions[0])
            print(state.shape)
            print(obs.shape)
            print(actions.shape)
            episode_reward += rew
        rewards.append(episode_reward)
        logger.record_tabular("Episode", episode)
        logger.record_tabular("Immediate Reward", float(episode_reward))
        logger.record_tabular("Running Average", float(np.mean(rewards)))
        logger.dump_tabular()
        episode += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--max_episodes", default="1500", type=int, help="Maximum number of episodes to play.")
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument("--load_model_path", help="Loading the saved model")

    global args
    args = parser.parse_args()    
    run()