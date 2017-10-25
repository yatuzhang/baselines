#!/usr/bin/env python

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger

def wrap_train(env):
    from baselines.common.atari_wrappers import (wrap_deepmind, FrameStack)
    env = wrap_deepmind(env, clip_rewards=True)
    env = FrameStack(env, 4)
    return env

def train(env_id, seed):
    from baselines.ppo1 import pposgd_simple, cnn_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank != 0: logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
    env = bench.Monitor(env, logger.get_dir() and 
        osp.join(logger.get_dir(), "%i.monitor.json" % rank))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    env = wrap_train(env)
    env.seed(workerseed)

    task_name = "ppo." + args.env.split("-")[0] + "." + ("%.2f"%args.entcoeff)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    pposgd_simple.learn(env, policy_fn, 
        max_timesteps=args.num_timesteps,
        timesteps_per_batch=256,
        clip_param=0.2, entcoeff=args.entcoeff,
        optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
        gamma=0.99, lam=0.95,
        sample_stochastic=args.sample_stochastic, task_name=task_name, save_per_iter=args.save_per_iter,
        ckpt_dir=args.checkpoint_dir, load_model_path=args.load_model_path, task=args.task)
    env.close()

def main():
    import argparse
    num_frames = 40e6
    num_timesteps = int(num_frames / 4 * 1.1)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--task', help='Choose to do which task', type=str, choices=['train', 'sample_trajectory', 'play'], default='train')
    parser.add_argument('--sample_stochastic', type=bool, default=False)
    parser.add_argument('--entcoeff', help='entropy coefficiency', type=float, default=0.01)
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=num_timesteps)
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    global args
    args = parser.parse_args()
    train(args.env, seed=args.seed)

if __name__ == '__main__':
    main()
