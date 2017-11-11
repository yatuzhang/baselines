# ACKTR

- Original paper: https://arxiv.org/abs/1708.05144
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- Need Python 3.5, Python 2.7 not supported
- Need to set environment variable OPENAI_LOGDIR so that it knows where to output the logs
- NOTE: the rewards output in terminal is the clipped reward, to see actual reward you need to go to the OPENAI_LOGDIR and find the monitor files
- `python run_atari.py --env "BreakoutNoFrameskip-v4"  --ckpt_dir "/some/dir/"` runs the algorithm for 40M frames = 10M timesteps on the Breakout Atari game.  Saves the model at the ckpt_dir every 100 timesteps
- `python sample_trajectory.py --load_model_path "/some/dir/checkpoint01200"` loads the saved model and samples the trajectories for a default of 1500 episodes (does not save the trajectories, currently the sampling part is integrated in the GAN training code, so that expert trajectories are generated on the fly.  This is just so that you can sample trajectories and save an animation)
- `python train_singletask_gan.py --load_model_path /some/dir --env "PongNoFrameskip-v4" --ckpt_dir /some/other/dir` trains a single task GAN from an trained expert saved at --load_model_path and saves the trained GAN model to --ckpt_dir
-  `python sample_trajectory_gan.py --load_model_dir /some/dir --env "PongNoFrameskip-v4"` plays the game using the trained single task GAN

Note: To get the same action space for all games, need to change self._action_set = self.ale.getMinimalActionSet() to self._action_set = self.ale.getLegalActionSet() in atari_env.py in the OpenAI gym files you installed
